# Standalone XGBoost training worker: one process per device, jobs run sequentially.
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

    import xgboost as xgb  # imported after CUDA_VISIBLE_DEVICES is set

    jobs = json.loads(open(args.jobs, "r", encoding="utf-8").read())
    cache = {}

    for job in jobs:
        tag = job["tag"]
        t0 = time.time()
        if job["data"] not in cache:
            cache[job["data"]] = pd.read_parquet(job["data"])
        df = cache[job["data"]]
        feats = job["features"]

        is_train = (df["role"] == "train").to_numpy()
        is_target = (df["role"] == "target").to_numpy()
        Xtr = df.loc[is_train, feats].to_numpy(dtype=np.float32)
        ytr = df.loc[is_train, "y"].to_numpy(dtype=np.float32)
        wtr = (
            df.loc[is_train, "w"].to_numpy(dtype=np.float32)
            if job["use_weights"]
            else None
        )

        params = dict(job["params"])
        params["device"] = device
        params["seed"] = int(job["seed"])
        if args.nthread:
            params["nthread"] = int(args.nthread)

        dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, weight=wtr, feature_names=feats)
        evals = [(dtrain, "train")]
        dvalid = None
        if job["evaluate"]:
            # Only open target rows carry a usable label; closed rows are deterministic zeros.
            mask = (
                is_target
                & (df["IsOpen"].to_numpy() == 1)
                & (df["OrderVolume"].to_numpy() > 0)
            )
            Xva = df.loc[mask, feats].to_numpy(dtype=np.float32)
            yva = df.loc[mask, "y"].to_numpy(dtype=np.float32)
            dvalid = xgb.QuantileDMatrix(
                Xva, label=yva, ref=dtrain, feature_names=feats
            )
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
        rng = (0, best_iter + 1)

        # predictions for every target row of this block, in the parquet's own row order
        Xall = df.loc[is_target, feats].to_numpy(dtype=np.float32)
        dall = xgb.DMatrix(Xall, feature_names=feats)
        preds = booster.predict(dall, iteration_range=rng)

        prefix = job["out_prefix"]
        booster.save_model(prefix + ".ubj")
        np.save(prefix + ".pred.npy", preds.astype(np.float32))
        score_map = booster.get_score(importance_type="gain")
        weight_map = booster.get_score(importance_type="weight")
        meta = {
            "tag": tag,
            "device": args.device,
            "seed": int(job["seed"]),
            "best_iteration": best_iter,
            "best_score": float(getattr(booster, "best_score", float("nan"))),
            "num_boost_round": total,
            "train_rows": int(is_train.sum()),
            "target_rows": int(is_target.sum()),
            "use_weights": bool(job["use_weights"]),
            "seconds": round(time.time() - t0, 1),
            "importance_gain": {k: float(v) for k, v in score_map.items()},
            "importance_weight": {k: float(v) for k, v in weight_map.items()},
            "params": {k: v for k, v in params.items()},
        }
        with open(prefix + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        emit(
            "DONE",
            {
                "tag": tag,
                "best_iteration": best_iter,
                "best_score": meta["best_score"],
                "seconds": meta["seconds"],
            },
        )

    emit("WORKER_DONE", {"device": args.device, "jobs": len(jobs)})


if __name__ == "__main__":
    main()
