"""Phase 3 + Phase 5 - do luong.

PLAN muc 5: bao cao macro-F1, per-class F1, FPR tren lop Benign, confusion
matrix. Khong dung accuracy tong vi benign (~4.039) it hon malware (~12.623).

Mot diem phuong phap quan trong o Phase 5: moi ky thuat obfuscation lam hong
mot so APK khac nhau, nen tap sau obf khong con giong tap sach. So F1 sach
tren 500 APK voi F1 obf tren 430 APK con lai la so hai thu khac nhau. Vi vay
moi Delta o day deu la so sanh GHEP CAP: F1 sach duoc tinh lai tren dung
nhung sha da song sot qua ky thuat do.

Chay:
    python -m src.evaluate --clean
    python -m src.evaluate --obf                  # tat ca ky thuat da co feature
    python -m src.evaluate --obf --techniques T3_string
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import config
from .features.vectorize import GroupVectorizer, load_feature_frame, parse_dicts
from .utils import atomic_write_json, read_json, read_sha_list, seed_stamp, setup_logging

log = setup_logging("evaluate")

MODEL_NAMES = ("rf", "xgb", "svm")
MODELS_INDEX = config.MODELS_DIR / "models_index.json"


# --------------------------------------------------------------------------
def model_path(featureset: str, model: str) -> Path:
    return config.MODELS_DIR / f"{featureset}__{model}.joblib"


def load_model(featureset: str, model: str):
    import joblib
    p = model_path(featureset, model)
    if not p.exists():
        raise FileNotFoundError(f"chua train model: {p}")
    return joblib.load(p)


def load_models_index() -> list[dict]:
    idx = read_json(MODELS_INDEX, default=None)
    if not idx:
        raise FileNotFoundError(f"chua co {MODELS_INDEX} - chay src.train truoc")
    return idx["models"]


# --------------------------------------------------------------------------
def decision_scores(clf, X) -> np.ndarray | None:
    """Diem lien tuc cho ROC-AUC. LinearSVC khong co predict_proba."""
    if hasattr(clf, "predict_proba"):
        try:
            return clf.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(clf, "decision_function"):
        try:
            return np.asarray(clf.decision_function(X)).ravel()
        except Exception:
            pass
    return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray | None = None,
                    categories: pd.Series | None = None) -> dict:
    """Nhan: 0 = Benign, 1 = Malware. Duong tinh = malware."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    m = {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_benign": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_malware": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_malware": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_malware": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        # FPR tren lop Benign: ti le benign bi goi nham la malware.
        "fpr_benign": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            pass

    if categories is not None:
        cats = np.asarray(categories)
        per_cat = {}
        for c in np.unique(cats):
            mask = cats == c
            if not mask.any():
                continue
            expected = 0 if str(c).lower() == config.BENIGN_CATEGORY.lower() else 1
            per_cat[str(c)] = {
                "n": int(mask.sum()),
                "recall": float((y_pred[mask] == expected).mean()),
            }
        m["per_category_recall"] = per_cat
    return m


# --------------------------------------------------------------------------
def _featureset_groups(featureset: str, vec: GroupVectorizer) -> list[str]:
    return vec.available_groups() if featureset == "ALL" else [featureset]


def evaluate_all_models(df: pd.DataFrame, vec: GroupVectorizer,
                        models: list[dict]) -> dict:
    """Chay moi model da train tren cung mot khung du lieu."""
    dicts = parse_dicts(df)
    X_full = vec.transform(dicts)
    y = df["label"].to_numpy(dtype=int)
    cats = df["category"] if "category" in df.columns else None

    out: dict[str, dict] = {}
    cache: dict[str, object] = {}
    for entry in models:
        fs, name = entry["featureset"], entry["model"]
        key = f"{fs}__{name}"
        if fs not in cache:
            cache[fs] = (X_full if fs == "ALL"
                         else vec.subset(X_full, _featureset_groups(fs, vec)))
        X = cache[fs]
        try:
            clf = load_model(fs, name)
        except FileNotFoundError as e:
            log.warning("bo qua %s: %s", key, e)
            continue
        y_pred = clf.predict(X)
        out[key] = compute_metrics(y, y_pred, decision_scores(clf, X), cats)
        out[key]["featureset"] = fs
        out[key]["model"] = name
    return out


# --------------------------------------------------------------------------
def evaluate_clean(split: str = "test") -> dict:
    vec = GroupVectorizer.load(config.MODELS_DIR / "vectorizer.joblib")
    models = load_models_index()

    df = load_feature_frame(config.CLEAN_FEATURE_PATH)
    shas = set(read_sha_list(config.SPLIT_DIR / f"{split}_sha256.txt"))
    df = df[df["sha256"].isin(shas)].reset_index(drop=True)
    log.info("Eval sach [%s]: %d APK (benign=%d)", split, len(df),
             int((df["label"] == 0).sum()))

    res = {
        **seed_stamp(),
        "split": split,
        "n_apk": len(df),
        "features_path": str(config.CLEAN_FEATURE_PATH),
        "vectorizer_n_features": vec.n_features_,
        "metrics": evaluate_all_models(df, vec, models),
    }
    out = config.RESULTS_DIR / ("clean_baseline.json" if split == "test"
                                else f"clean_{split}.json")
    atomic_write_json(out, res)
    log.info("-> %s", out)
    _print_table(res["metrics"], f"SACH [{split}]")
    return res


def evaluate_obfuscated(techniques: list[str]) -> dict:
    vec = GroupVectorizer.load(config.MODELS_DIR / "vectorizer.joblib")
    models = load_models_index()
    clean_df = load_feature_frame(config.CLEAN_FEATURE_PATH)

    all_res = {}
    for tech in techniques:
        path = config.technique_feature_path(tech)
        if not path.exists():
            log.warning("[%s] chua co feature (%s) - bo qua", tech, path.name)
            continue

        obf_df = load_feature_frame(path).reset_index(drop=True)
        survived = set(obf_df["sha256"])

        # So sanh ghep cap: tinh lai F1 sach tren DUNG nhung sha song sot.
        paired_clean = clean_df[clean_df["sha256"].isin(survived)].reset_index(drop=True)
        obf_df = obf_df[obf_df["sha256"].isin(set(paired_clean["sha256"]))].reset_index(drop=True)

        # Nhan cua ban obf lay tu ban sach - obfuscation khong doi nhan.
        if "label" not in obf_df.columns or obf_df["label"].isna().any():
            obf_df = obf_df.drop(columns=[c for c in ("label", "category")
                                          if c in obf_df.columns])
            obf_df = obf_df.merge(clean_df[["sha256", "label", "category"]],
                                  on="sha256", how="left")

        obf_df = obf_df.sort_values("sha256").reset_index(drop=True)
        paired_clean = paired_clean.sort_values("sha256").reset_index(drop=True)

        log.info("[%s] %d APK ghep cap duoc", tech, len(obf_df))
        res = {
            **seed_stamp(),
            "technique": tech,
            "obfuscators": config.TECHNIQUES.get(tech, []),
            "n_paired": len(obf_df),
            "metrics_obf": evaluate_all_models(obf_df, vec, models),
            "metrics_clean_paired": evaluate_all_models(paired_clean, vec, models),
        }
        out = config.RESULTS_DIR / f"obf_{tech}.json"
        atomic_write_json(out, res)
        log.info("-> %s", out)
        _print_delta_table(res, tech)
        all_res[tech] = res
    return all_res


def _print_table(metrics: dict, title: str) -> None:
    rows = [{"model": k, "macro_f1": round(v["macro_f1"], 4),
             "f1_benign": round(v["f1_benign"], 4),
             "f1_malware": round(v["f1_malware"], 4),
             "fpr_benign": round(v["fpr_benign"], 4),
             "roc_auc": round(v.get("roc_auc", float("nan")), 4)}
            for k, v in sorted(metrics.items())]
    if rows:
        print(f"\n=== {title} ===")
        print(pd.DataFrame(rows).to_string(index=False))


def _print_delta_table(res: dict, tech: str) -> None:
    rows = []
    for k, obf in sorted(res["metrics_obf"].items()):
        cln = res["metrics_clean_paired"].get(k)
        if not cln:
            continue
        rows.append({
            "model": k,
            "f1_clean": round(cln["macro_f1"], 4),
            "f1_obf": round(obf["macro_f1"], 4),
            "delta": round(obf["macro_f1"] - cln["macro_f1"], 4),
            "fpr_benign_obf": round(obf["fpr_benign"], 4),
        })
    if rows:
        print(f"\n=== {tech} (n={res['n_paired']}) ===")
        print(pd.DataFrame(rows).to_string(index=False))


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3/5 - danh gia")
    ap.add_argument("--clean", action="store_true", help="eval tren tap test sach")
    ap.add_argument("--val", action="store_true", help="eval tren tap val")
    ap.add_argument("--obf", action="store_true", help="eval tren APK da obfuscate")
    ap.add_argument("--techniques", nargs="*", default=None)
    args = ap.parse_args()

    config.ensure_dirs()
    if not (args.clean or args.obf or args.val):
        args.clean = True

    if args.val:
        evaluate_clean("val")
    if args.clean:
        evaluate_clean("test")
    if args.obf:
        techs = args.techniques or [t for t in config.TECHNIQUE_ORDER
                                    if config.technique_feature_path(t).exists()]
        if not techs:
            log.error("Khong co feature obfuscated nao. Chay src.obfuscate roi "
                      "src.features.extract --tag <technique> truoc.")
            return
        evaluate_obfuscated(techs)


if __name__ == "__main__":
    main()
