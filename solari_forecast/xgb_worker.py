# Standalone GPU training worker (XGBoost or CatBoost). The dispatcher starts one process per
# job and pins it to one card, so several jobs can share a GPU without sharing a CUDA context.
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd


def emit(kind, payload):
    sys.stdout.write(kind + " " + json.dumps(payload) + chr(10))
    sys.stdout.flush()


def make_callback(xgb, tag, total, every=25):
    class _CB(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if epoch % every == 0 or epoch == total - 1:
                metrics = {}
                for split, mdict in evals_log.items():
                    for metric, values in mdict.items():
                        metrics[split + "-" + metric] = float(values[-1])
                emit(
                    "PROGRESS",
                    {"tag": tag, "iter": epoch + 1, "total": total, "metrics": metrics},
                )
            return False

    return _CB()


def row_masks(df, job):
    if "train_end" in job:
        # direct-model table: every observed hub-day; the job names its own windows
        dates = df["Date"].to_numpy()
        good = (df["IsOpen"].to_numpy() == 1) & (df["OrderVolume"].to_numpy() > 0)
        is_train = good & (dates <= np.datetime64(job["train_end"]))
        is_target = (dates >= np.datetime64(job["target_start"])) & (
            dates <= np.datetime64(job["target_end"])
        )
    else:
        is_train = (df["role"] == "train").to_numpy()
        is_target = (df["role"] == "target").to_numpy()
    # only open target rows carry a usable label; closed rows are deterministic zeros
    is_eval = (
        is_target & (df["IsOpen"].to_numpy() == 1) & (df["OrderVolume"].to_numpy() > 0)
    )
    if job.get("eval_testlike"):
        # the test window has no regional holiday: stop on rows that look like it
        is_eval &= (df["RegionalHoliday"].to_numpy() == 0) & (
            df["HolidaysThisWeek"].to_numpy() == 0
        )
    return is_train, is_target, is_eval


def train_xgboost(job, df, masks, device, nthread):
    import xgboost as xgb  # imported after CUDA_VISIBLE_DEVICES is set

    is_train, is_target, is_eval = masks
    tag, feats = job["tag"], job["features"]
    cats = set(job.get("categorical", []))
    ftypes = ["c" if f in cats else "q" for f in feats]
    dm_kw = {
        "feature_names": feats,
        "feature_types": ftypes,
        "enable_categorical": bool(cats),
    }

    Xtr = df.loc[is_train, feats].to_numpy(dtype=np.float32)
    ytr = df.loc[is_train, "y"].to_numpy(dtype=np.float32)
    wtr = (
        df.loc[is_train, "w"].to_numpy(dtype=np.float32) if job["use_weights"] else None
    )

    params = dict(job["params"])
    params["device"] = device
    params["seed"] = int(job["seed"])
    # start from the (weighted) mean log volume: a robust loss would otherwise need hundreds
    # of rounds just to walk from the default intercept to ~8.7
    params.setdefault("base_score", float(np.average(ytr, weights=wtr)))
    if nthread:
        params["nthread"] = int(nthread)

    dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, weight=wtr, **dm_kw)
    evals = [(dtrain, "train")]
    if job["evaluate"]:
        Xva = df.loc[is_eval, feats].to_numpy(dtype=np.float32)
        yva = df.loc[is_eval, "y"].to_numpy(dtype=np.float32)
        dvalid = xgb.QuantileDMatrix(Xva, label=yva, ref=dtrain, **dm_kw)
        evals.append((dvalid, "valid"))

    total = int(job["num_boost_round"])
    callbacks = [make_callback(xgb, tag, total, job.get("progress_every", 25))]
    kwargs = {}
    if job["evaluate"] and job.get("early_stopping"):
        kwargs["early_stopping_rounds"] = int(job["early_stopping"])

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=total,
        evals=evals,
        verbose_eval=False,
        callbacks=callbacks,
        **kwargs
    )
    best_iter = int(getattr(booster, "best_iteration", total - 1))
    best_score = float(getattr(booster, "best_score", float("nan")))

    # predictions for every target row of this block, in the parquet's own row order
    dall = xgb.DMatrix(df.loc[is_target, feats].to_numpy(dtype=np.float32), **dm_kw)
    preds = booster.predict(dall, iteration_range=(0, best_iter + 1))

    booster.save_model(job["out_prefix"] + ".ubj")
    importance = {
        "importance_gain": {
            k: float(v) for k, v in booster.get_score(importance_type="gain").items()
        },
        "importance_weight": {
            k: float(v) for k, v in booster.get_score(importance_type="weight").items()
        },
    }
    return preds, best_iter, best_score, importance, params


def train_catboost(job, df, masks, device, nthread):
    from catboost import CatBoostRegressor, Pool

    is_train, is_target, is_eval = masks
    tag, feats = job["tag"], job["features"]
    cats = list(job.get("categorical", []))

    def frame(mask):
        X = df.loc[mask, feats].copy()
        for c in cats:
            X[c] = X[c].astype("int64")  # CatBoost needs integer (or string) categories
        return X

    ytr = df.loc[is_train, "y"].to_numpy(dtype=np.float64)
    wtr = (
        df.loc[is_train, "w"].to_numpy(dtype=np.float64) if job["use_weights"] else None
    )
    ptrain = Pool(frame(is_train), label=ytr, weight=wtr, cat_features=cats)

    total = int(job["num_boost_round"])
    early = (
        int(job["early_stopping"])
        if job["evaluate"] and job.get("early_stopping")
        else 0
    )
    params = dict(job["params"])
    params.update(
        iterations=total,
        random_seed=int(job["seed"]),
        task_type="GPU" if device != "cpu" else "CPU",
        verbose=0,
        allow_writing_files=False,
        use_best_model=bool(early),  # never pick rounds on a block that is being scored
    )
    if device != "cpu":
        params["devices"] = "0"
    else:
        params.pop("gpu_ram_part", None)
    if nthread:
        params["thread_count"] = int(nthread)

    fit_kw = {}
    if early:
        yva = df.loc[is_eval, "y"].to_numpy(dtype=np.float64)
        fit_kw["eval_set"] = Pool(frame(is_eval), label=yva, cat_features=cats)
        fit_kw["early_stopping_rounds"] = early
    emit("PROGRESS", {"tag": tag, "iter": 0, "total": total, "metrics": {}})
    model = CatBoostRegressor(**params)
    model.fit(ptrain, **fit_kw)

    best_iter = model.get_best_iteration() if early else None
    best_iter = int(best_iter) if best_iter is not None else int(model.tree_count_) - 1
    best_score = float("nan")
    if early:
        best_score = float(
            model.get_best_score().get("validation", {}).get("RMSE", float("nan"))
        )
    emit(
        "PROGRESS",
        {
            "tag": tag,
            "iter": best_iter + 1,
            "total": total,
            "metrics": {"valid-rmse": best_score} if early else {},
        },
    )

    preds = model.predict(frame(is_target))
    model.save_model(job["out_prefix"] + ".cbm")
    importance = {
        "importance_gain": {
            f: float(v) for f, v in zip(feats, model.get_feature_importance())
        },
    }
    return preds, best_iter, best_score, importance, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="JSON file holding the list of jobs")
    ap.add_argument("--device", required=True, help="cuda:N or cpu")
    ap.add_argument("--nthread", type=int, default=0)
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        # Pin this process to exactly one physical GPU; it then addresses it as cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]
        device = "cuda:0"
    else:
        device = "cpu"

    jobs = json.loads(open(args.jobs, "r", encoding="utf-8").read())
    cache = {}

    for job in jobs:
        tag = job["tag"]
        t0 = time.time()
        if job["data"] not in cache:
            cache.clear()  # one table at a time bounds worker RAM
            cache[job["data"]] = pd.read_parquet(job["data"])
        df = cache[job["data"]]
        masks = row_masks(df, job)

        trainer = train_catboost if job.get("lib") == "catboost" else train_xgboost
        preds, best_iter, best_score, importance, params = trainer(
            job, df, masks, device, args.nthread
        )

        prefix = job["out_prefix"]
        np.save(prefix + ".pred.npy", np.asarray(preds, dtype=np.float32))
        meta = {
            "tag": tag,
            "lib": job.get("lib", "xgboost"),
            "device": args.device,
            "seed": int(job["seed"]),
            "best_iteration": best_iter,
            "best_score": best_score,
            "num_boost_round": int(job["num_boost_round"]),
            "train_rows": int(masks[0].sum()),
            "target_rows": int(masks[1].sum()),
            "use_weights": bool(job["use_weights"]),
            "seconds": round(time.time() - t0, 1),
            **importance,
            "params": {k: v for k, v in params.items()},
        }
        with open(prefix + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)
        emit(
            "DONE",
            {
                "tag": tag,
                "best_iteration": best_iter,
                "best_score": best_score,
                "seconds": meta["seconds"],
            },
        )

    emit("WORKER_DONE", {"device": args.device, "jobs": len(jobs)})


if __name__ == "__main__":
    main()
