#!/usr/bin/env python3
"""F4 real-data trust gate: validate fingerspelling abstention on held-out set C (Gate F4).

F2 calibrated per-letter thresholds ON the held-out set (small n, self-described directional).
This gate closes that loop honestly:
  (a) score the CURRENT thresholds on set_c (overall + per-letter selective precision/coverage)
  (b) RECALIBRATE on set_a+set_b only (never touching set_c), validate on set_c
  (c) per-pair confusion analysis for the pinned letters (threshold >= 0.99 -> never fire)
  (d) CONFUSABLE-CLUSTER policy: instead of suppressing pinned letters silently, emit a
      distinct "spell again" prompt at ONE shared threshold calibrated on set_a+set_b —
      the prompt counts as answered, and it is correct iff the true letter is in the cluster
  (e) write checkpoints/f4_gate_report.json with all numbers + a verdict against the bar:
      selective precision >= 0.95 target, >= 0.90 floor, at stated coverage

abstention.json is only rewritten when a challenger policy validates better on set_c
(pass --no-write to suppress even then). Schema stays backward-compatible: recognizer.py
reads only global_threshold + per_letter_threshold; the cluster policy is an ADDITIVE key.

Usage (from repo root, after `source env.sh`):
  python models/fingerspelling/f4_gate.py \
    --model models/fingerspelling/checkpoints/mlp.pt \
    --calib data/fingerspelling/set_a.npz data/fingerspelling/set_b.npz \
    --heldout data/fingerspelling/set_c.npz \
    --abstention models/fingerspelling/checkpoints/abstention.json \
    --out models/fingerspelling/checkpoints/f4_gate_report.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

# eval/ is on PYTHONPATH via env.sh
from abstention import selective_metrics

from models.fingerspelling.calibrate import items_from_report, per_letter_thresholds
from models.fingerspelling.eval_fingerspelling import evaluate

PIN_EPS = 0.99                       # threshold >= this ~= "letter can never fire"
CLUSTER_LABEL = "confusable_cluster"
CLUSTER_REASON = "confusable — please spell again slowly"
TARGET_PRECISION = 0.95
FLOOR_PRECISION = 0.90


class PerLetterPolicy:
    """Duck-types eval/abstention.AbstentionPolicy for selective_metrics(): accept the top-1
    letter iff its score clears that letter's threshold (mirrors recognizer._threshold_for /
    decide_from_probs — a threshold >= 1.0 pins the letter shut)."""

    def __init__(self, per_letter, default_threshold, name="per_letter"):
        self.per_letter = per_letter
        self.default_threshold = default_threshold
        self.conf_threshold = name     # echoed into selective_metrics rows (informational)
        self.margin_threshold = 0.0

    def decide(self, candidates):
        if not candidates:
            return None, "no_scores"
        label, score = candidates[0]
        thr = self.per_letter.get(label, self.default_threshold)
        if score >= thr and thr < 1.0:
            return label, "accept"
        return None, f"abstain(conf={score:.2f}<thr={thr:.2f})"


class ClusterPolicy(PerLetterPolicy):
    """Per-letter policy plus a confusable super-class: when top-1 is a cluster letter and
    clears the SHARED cluster threshold, answer the '{CLUSTER_REASON}' prompt instead of a
    letter. Score it against expected labels remapped by cluster_expected()."""

    def __init__(self, per_letter, cluster, cluster_threshold, default_threshold):
        super().__init__(per_letter, default_threshold, name="per_letter+cluster")
        self.cluster = frozenset(cluster)
        self.cluster_threshold = cluster_threshold

    def decide(self, candidates):
        if not candidates:
            return None, "no_scores"
        label, score = candidates[0]
        if label in self.cluster:
            if score >= self.cluster_threshold:
                return CLUSTER_LABEL, "confusable"
            return None, f"abstain(conf={score:.2f}<cluster_thr={self.cluster_threshold:.2f})"
        return super().decide(candidates)


def model_probs(model_path, npz_paths):
    """Softmax probs + labels for one or more feature sets (same plumbing as eval_fingerspelling)."""
    import torch
    from models.fingerspelling.features import augment_features
    from models.fingerspelling.train_fingerspelling import MLP

    Xs, ys = [], []
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        Xs.append(d["X"].astype(np.float32)); ys.append(d["y"].astype(np.int64))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)  # trusted local ckpt
    if ckpt.get("geometric"):
        X = augment_features(X)
        X = ((X - ckpt["mean"]) / ckpt["std"]).astype(np.float32)
    model = MLP(ckpt["input_dim"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X)), dim=1).numpy()
    return probs, y


def cluster_expected(items, cluster):
    """Remap expected labels into the cluster super-class so selective_metrics scores the
    'spell again' prompt as correct iff the signer's true letter is in the cluster."""
    return [{"candidates": it["candidates"],
             "expected": CLUSTER_LABEL if it["expected"] in cluster else it["expected"]}
            for it in items]


def calibrate_cluster_threshold(results, cluster, target_precision):
    """Shared threshold for the cluster prompt, via per_letter_thresholds on remapped rows
    (reuse — the pooled cluster is just one 'letter' to the calibrator)."""
    pseudo = [{"expected": CLUSTER_LABEL if r["expected"] in cluster else r["expected"],
               "topk": [{"label": CLUSTER_LABEL, "score": r["topk"][0]["score"]}]}
              for r in results if r.get("topk") and r["topk"][0]["label"] in cluster]
    thresholds, diag = per_letter_thresholds(pseudo, target_precision)
    return thresholds.get(CLUSTER_LABEL, 1.0), diag.get(CLUSTER_LABEL, {"note": "cluster never predicted"})


def per_letter_breakdown(items, policy):
    """Per-letter view of one policy: coverage of each TRUE letter + precision of each
    EMITTED label (decision logic reused from the policy — no metric math duplicated)."""
    by_true, by_pred = {}, {}
    for it in items:
        label, _reason = policy.decide(it["candidates"])
        row = by_true.setdefault(it["expected"], {"n_true": 0, "answered": 0})
        row["n_true"] += 1
        if label is None:
            continue
        row["answered"] += 1
        p = by_pred.setdefault(label, {"n_emitted": 0, "correct": 0})
        p["n_emitted"] += 1
        if label == it["expected"]:
            p["correct"] += 1
    out = {}
    for letter in sorted(set(by_true) | set(by_pred)):
        t = by_true.get(letter, {"n_true": 0, "answered": 0})
        p = by_pred.get(letter, {"n_emitted": 0, "correct": 0})
        out[letter] = {
            "n_true": t["n_true"],
            "coverage_of_true": round(t["answered"] / t["n_true"], 4) if t["n_true"] else None,
            "n_emitted": p["n_emitted"],
            "emission_precision": round(p["correct"] / p["n_emitted"], 4) if p["n_emitted"] else None,
        }
    return out


def confusion_analysis(results, letters, cluster):
    """For each pinned letter on the held-out set: what it is predicted AS (as_true), what
    truly underlies its predictions (as_predicted), with counts + mean top-1 confidence."""

    def fmt(d):
        return {k: {"count": v["count"], "mean_conf": round(float(np.mean(v["confs"])), 3)}
                for k, v in sorted(d.items(), key=lambda kv: -kv[1]["count"])}

    out = {}
    for letter in sorted(letters):
        as_true, as_pred = {}, {}
        for r in results:
            top = r["topk"][0]
            if r["expected"] == letter:
                d = as_true.setdefault(top["label"], {"count": 0, "confs": []})
                d["count"] += 1; d["confs"].append(top["score"])
            if top["label"] == letter:
                d = as_pred.setdefault(r["expected"], {"count": 0, "confs": []})
                d["count"] += 1; d["confs"].append(top["score"])
        n_pred = sum(v["count"] for v in as_pred.values())
        within = sum(v["count"] for k, v in as_pred.items() if k in cluster)
        out[letter] = {"as_true": fmt(as_true), "as_predicted": fmt(as_pred), "n_pred": n_pred,
                       "within_cluster_rate": round(within / n_pred, 3) if n_pred else None}
    return out


def choose_policy(overall, target_precision, floor_precision):
    """Best validated policy on set_c: highest coverage at target precision; else highest
    coverage at the floor; else keep the status quo (do-no-harm default)."""
    def precision(name):
        p = overall[name]["selective_accuracy"]
        return p if p is not None else 0.0
    for bar in (target_precision, floor_precision):
        ok = [name for name in overall if precision(name) >= bar]
        if ok:
            return max(ok, key=lambda name: overall[name]["coverage"])
    return "status_quo"


def verdict_for(row, target_precision, floor_precision):
    p = row["selective_accuracy"] if row["selective_accuracy"] is not None else 0.0
    if row["coverage"] == 0:
        return "fail"
    if p >= target_precision:
        return "pass"
    if p >= floor_precision:
        return "conditional"
    return "fail"


def update_abstention(cfg, chosen, recal, recal_diag, cluster, cluster_tau, cluster_metrics):
    """Return the new abstention.json content for a validated challenger (schema-compatible:
    recognizer.py reads only global_threshold + per_letter_threshold)."""
    new = dict(cfg)
    if chosen == "recalibrated":
        new["per_letter_threshold"] = recal
        new["diagnostics"] = recal_diag
        new["note"] = "F4: recalibrated on set_a+set_b only; validated on held-out set_c"
    elif chosen == "cluster_policy":
        new[CLUSTER_LABEL] = {
            "letters": sorted(cluster), "threshold": cluster_tau, "reason": CLUSTER_REASON,
            "calibrated_on": "set_a+set_b", "validated_on_set_c": cluster_metrics,
        }
        new["note"] = ("F4: per-letter thresholds unchanged (validated on set_c); pinned letters "
                       "additionally emit the confusable-cluster prompt — additive key, "
                       "readers of per_letter_threshold are unaffected")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/fingerspelling/checkpoints/mlp.pt")
    ap.add_argument("--calib", nargs="+",
                    default=["data/fingerspelling/set_a.npz", "data/fingerspelling/set_b.npz"])
    ap.add_argument("--heldout", default="data/fingerspelling/set_c.npz")
    ap.add_argument("--abstention", default="models/fingerspelling/checkpoints/abstention.json")
    ap.add_argument("--out", default="models/fingerspelling/checkpoints/f4_gate_report.json")
    ap.add_argument("--target-precision", type=float, default=TARGET_PRECISION)
    ap.add_argument("--floor-precision", type=float, default=FLOOR_PRECISION)
    ap.add_argument("--no-write", action="store_true",
                    help="Never rewrite abstention.json, even for a validated challenger.")
    args = ap.parse_args()

    cfg = json.loads(Path(args.abstention).read_text(encoding="utf-8"))
    current = {k: float(v) for k, v in cfg["per_letter_threshold"].items()}
    default_thr = float(cfg.get("global_threshold", 1.0))
    cluster = sorted(k for k, v in current.items() if v >= PIN_EPS)

    # Model probs -> report-shaped results (reuse eval_fingerspelling.evaluate)
    probs_ab, y_ab = model_probs(args.model, args.calib)
    probs_c, y_c = model_probs(args.model, [args.heldout])
    results_ab = evaluate(probs_ab, y_ab)["results"]
    results_c = evaluate(probs_c, y_c)["results"]
    items_c = items_from_report(results_c)

    # (a) status quo on held-out C  — caveat: F2 fit these thresholds ON set_c (in-sample)
    sq = PerLetterPolicy(current, default_thr, name="status_quo")

    # (b) recalibrate per-letter on set_a+set_b only (the model's TRAIN sets — no clean
    # calibration split exists; expect optimistic thresholds, judged on set_c)
    recal, recal_diag = per_letter_thresholds(results_ab, args.target_precision)
    re = PerLetterPolicy(recal, 1.0, name="recalibrated")

    # (c) per-pair confusion for pinned letters on held-out C
    confusion = confusion_analysis(results_c, cluster, set(cluster))

    # (d) confusable-cluster policy: shared threshold from set_a+set_b, scored on C
    cluster_tau, cluster_diag = calibrate_cluster_threshold(results_ab, set(cluster),
                                                            args.target_precision)
    cl = ClusterPolicy(current, cluster, cluster_tau, default_thr)
    items_c_cluster = cluster_expected(items_c, set(cluster))

    overall = {
        "status_quo": selective_metrics(items_c, sq),
        "recalibrated": selective_metrics(items_c, re),
        "cluster_policy": selective_metrics(items_c_cluster, cl),
    }
    per_letter = {
        "status_quo": per_letter_breakdown(items_c, sq),
        "recalibrated": per_letter_breakdown(items_c, re),
        "cluster_policy": per_letter_breakdown(items_c_cluster, cl),
    }

    chosen = choose_policy(overall, args.target_precision, args.floor_precision)
    verdict = verdict_for(overall[chosen], args.target_precision, args.floor_precision)

    updated = False
    if chosen != "status_quo" and not args.no_write:
        new_cfg = update_abstention(cfg, chosen, recal, recal_diag, cluster, cluster_tau,
                                    overall["cluster_policy"])
        Path(args.abstention).write_text(json.dumps(new_cfg, indent=2), encoding="utf-8")
        updated = True
        print(f"updated {args.abstention} (policy: {chosen})")

    report = {
        "gate": "F4",
        "generated_by": "models/fingerspelling/f4_gate.py",
        "model": args.model,
        "calibration_data": args.calib,
        "heldout_data": args.heldout,
        "n_heldout": len(items_c),
        "target_precision": args.target_precision,
        "floor_precision": args.floor_precision,
        "cluster": {"letters": cluster, "threshold": cluster_tau, "reason": CLUSTER_REASON,
                    "label": CLUSTER_LABEL, "calibration_diagnostics": cluster_diag},
        "recalibrated_thresholds": recal,
        "policies": {name: {"overall": overall[name], "per_letter": per_letter[name]}
                     for name in overall},
        "policy_notes": {
            "status_quo": "per-letter thresholds from F2 were calibrated ON set_c (in-sample "
                          "optimistic); pinned letters never fire",
            "recalibrated": "thresholds fit on set_a+set_b = the MLP's training data "
                            "(overconfident there); this row is the honest held-out check",
            "cluster_policy": "letter emissions identical to status_quo; pinned letters emit "
                              "the spell-again prompt, correct iff true letter is in-cluster",
        },
        "confusion_analysis": confusion,
        "chosen_policy": chosen,
        "abstention_json_updated": updated,
        "verdict": verdict,
        "verdict_detail": (f"{chosen} on held-out {Path(args.heldout).name}: selective precision "
                           f"{overall[chosen]['selective_accuracy']}, coverage "
                           f"{overall[chosen]['coverage']} (target {args.target_precision}, "
                           f"floor {args.floor_precision})"),
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"pinned letters (thr>={PIN_EPS}): {cluster}   shared cluster tau={cluster_tau}")
    print("held-out set_c (n={}):".format(len(items_c)))
    for name, row in overall.items():
        print(f"  {name:14s} coverage={row['coverage']:.3f}  selective_precision="
              f"{row['selective_accuracy']}  confident_wrong={row['confident_wrong_rate']:.4f}")
    print(f"chosen={chosen}  verdict={verdict}  abstention.json updated={updated}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
