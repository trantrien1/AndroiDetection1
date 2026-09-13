"""Phase 3 - baseline ML (PLAN muc 5).

Day la baseline de so, khong phai de khoe: khong tuning hyperparameter.
Ba model - Random Forest (n=300), XGBoost, Linear SVM.

Ngoai model tren toan bo feature, module nay con train MOT MODEL CHO MOI NHOM
FEATURE rieng le (G1 mot minh, G2 mot minh, ...). Khong co nhung model do thi
Bang B o Phase 5 khong ton tai, va cung khong tra loi duoc cau hoi "nhom nao
dang ganh model".

Vectorizer duoc fit CHI tren train. Sai o day la leakage.

Chay:
    python -m src.train
    python -m src.train --csv-baseline path/to/cic_features.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from . import config
from .evaluate import MODEL_NAMES, compute_metrics, decision_scores, model_path
from .features.vectorize import GroupVectorizer, load_feature_frame, parse_dicts
from .utils import atomic_write_json, read_sha_list, seed_stamp, set_seed, setup_logging, timed

log = setup_logging("train")


# --------------------------------------------------------------------------
# Model
#
# class_weight='balanced' khong phai tuning de day so, ma de lop Benign
# (~4.039 so voi ~12.623 malware) khong bi nghien nat - von la ly do PLAN
# yeu cau bao cao macro-F1 thay vi accuracy tong. Tat bang --no-class-weight.
# --------------------------------------------------------------------------
def build_model(name: str, class_weight: str | None, seed: int):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=seed,
            class_weight=("balanced_subsample" if class_weight else None))

    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            log.warning("khong co xgboost - bo qua model 'xgb'")
            return None
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.3,
            tree_method="hist", n_jobs=-1, random_state=seed,
            eval_metric="logloss", verbosity=0)

    if name == "svm":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import MaxAbsScaler
        from sklearn.svm import LinearSVC
        # MaxAbsScaler giu nguyen tinh thua cua ma tran; StandardScaler thi khong.
        try:
            svc = LinearSVC(C=1.0, random_state=seed, max_iter=5000,
                            class_weight=class_weight, dual="auto")
        except TypeError:      # sklearn cu chua co dual='auto'
            svc = LinearSVC(C=1.0, random_state=seed, max_iter=5000,
                            class_weight=class_weight)
        return make_pipeline(MaxAbsScaler(), svc)

    raise ValueError(f"model khong biet: {name}")


def _pos_weight(y: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return (n_neg / n_pos) if n_pos else 1.0


def fit_one(name: str, X, y, class_weight: str | None, seed: int):
    clf = build_model(name, class_weight, seed)
    if clf is None:
        return None
    if name == "xgb" and class_weight:
        clf.set_params(scale_pos_weight=_pos_weight(y))
    if name == "xgb" and sparse.issparse(X):
        X = X.tocsr()
    clf.fit(X, y)
    return clf


# --------------------------------------------------------------------------
def train_all(features_path: Path, models: list[str], per_group_models: list[str],
              min_df: int, max_per_group: int, groups: list[str],
              class_weight: str | None, seed: int) -> dict:
    set_seed(seed)
    import joblib

    df = load_feature_frame(features_path)
    split_shas = {name: set(read_sha_list(config.SPLIT_DIR / f"{name}_sha256.txt"))
                  for name in ("train", "val", "test")}

    parts = {}
    for name, shas in split_shas.items():
        sub = df[df["sha256"].isin(shas)].reset_index(drop=True)
        parts[name] = sub
        log.info("%-5s %5d APK (benign=%d)", name, len(sub), int((sub["label"] == 0).sum()))
    if parts["train"].empty:
        raise RuntimeError("tap train rong - kiem tra lai da chay src.split va "
                           "src.features.extract chua")

    # --- fit vectorizer CHI tren train -----------------------------------
    with timed(log, "fit vectorizer (chi tren train)"):
        vec = GroupVectorizer(groups=groups, min_df=min_df,
                              max_per_group=max_per_group)
        vec.fit(parse_dicts(parts["train"]))
    vec.save(config.MODELS_DIR / "vectorizer.joblib")

    X = {name: vec.transform(parse_dicts(p)) for name, p in parts.items()}
    y = {name: p["label"].to_numpy(dtype=int) for name, p in parts.items()}

    featuresets: list[tuple[str, list[str]]] = [("ALL", vec.available_groups())]
    featuresets += [(g, [g]) for g in vec.available_groups()]

    index: list[dict] = []
    summary: dict[str, dict] = {}

    for fs, fs_groups in featuresets:
        Xs = {k: (v if fs == "ALL" else vec.subset(v, fs_groups)) for k, v in X.items()}
        want = models if fs == "ALL" else per_group_models
        for name in want:
            t0 = time.time()
            clf = fit_one(name, Xs["train"], y["train"], class_weight, seed)
            if clf is None:
                continue
            train_s = time.time() - t0

            key = f"{fs}__{name}"
            metrics = {}
            for split in ("val", "test"):
                if len(y[split]) == 0:
                    continue
                pred = clf.predict(Xs[split])
                metrics[split] = compute_metrics(
                    y[split], pred, decision_scores(clf, Xs[split]),
                    parts[split].get("category"))

            p = model_path(fs, name)
            p.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, p, compress=3)

            index.append({"featureset": fs, "model": name, "groups": fs_groups,
                          "path": str(p), "n_features": Xs["train"].shape[1],
                          "train_seconds": round(train_s, 1)})
            summary[key] = {"n_features": Xs["train"].shape[1],
                            "train_seconds": round(train_s, 1), **metrics}
            log.info("%-12s n_feat=%-7d train=%5.1fs | val macroF1=%.4f  test macroF1=%.4f",
                     key, Xs["train"].shape[1], train_s,
                     metrics.get("val", {}).get("macro_f1", float("nan")),
                     metrics.get("test", {}).get("macro_f1", float("nan")))

    atomic_write_json(config.MODELS_DIR / "models_index.json",
                      {**seed_stamp(), "models": index})

    out = {
        **seed_stamp(),
        "features_path": str(features_path),
        "n_train": len(parts["train"]), "n_val": len(parts["val"]),
        "n_test": len(parts["test"]),
        "vectorizer": {"n_features": vec.n_features_, "min_df": min_df,
                       "max_per_group": max_per_group,
                       "groups": vec.available_groups(),
                       "group_sizes": {g: int(vec.group_slices_[g][1] - vec.group_slices_[g][0])
                                       for g in vec.available_groups()}},
        "class_weight": class_weight,
        "results": summary,
    }
    atomic_write_json(config.RESULTS_DIR / "train_summary.json", out)
    _dump_top_features(vec, seed)
    _print_group_table(summary)
    return out


def _dump_top_features(vec: GroupVectorizer, seed: int) -> None:
    """Top feature cua RF toan bo - doc kem Bang B thi biet nhom nao dang ganh."""
    try:
        import joblib
        clf = joblib.load(model_path("ALL", "rf"))
        imp = getattr(clf, "feature_importances_", None)
        if imp is None:
            return
        order = np.argsort(imp)[::-1][:50]
        top = [{"feature": vec.feature_names_[i], "importance": float(imp[i])}
               for i in order]
        by_group: dict[str, float] = {}
        for i, v in enumerate(imp):
            g = vec.feature_names_[i].split(":", 1)[0]
            by_group[g] = by_group.get(g, 0.0) + float(v)
        atomic_write_json(config.RESULTS_DIR / "top_features.json",
                          {**seed_stamp(), "top50": top,
                           "importance_mass_by_group": by_group})
    except Exception as e:
        log.debug("bo qua top feature: %s", e)


def _print_group_table(summary: dict) -> None:
    rows = []
    for key, v in sorted(summary.items()):
        t = v.get("test", {})
        rows.append({"featureset__model": key, "n_features": v["n_features"],
                     "test_macro_f1": round(t.get("macro_f1", float("nan")), 4),
                     "test_fpr_benign": round(t.get("fpr_benign", float("nan")), 4)})
    if rows:
        print("\n=== Phase 3: baseline sach ===")
        print(pd.DataFrame(rows).to_string(index=False))


# --------------------------------------------------------------------------
# Baseline doi chieu tren CSV goc cua CIC (PLAN muc 4)
#
# Bo CSV nay khong dung duoc cho phan obfuscation (APK da obfuscate khong co
# dong CSV tuong ung), nhung day la bo ma phan lon paper tren CICMalDroid dung,
# nen con so sach o day so sanh truc tiep duoc voi literature. Bao cao ca hai.
#
# CSV cua CIC khong co cot sha256 nen KHONG ghep duoc theo tung mau voi split
# o tren. Day la mot baseline doc lap, chi so sanh duoc ve mat phan phoi.
# --------------------------------------------------------------------------
def train_csv_baseline(csv_path: Path, models: list[str], class_weight: str | None,
                       seed: int, label_col: str | None = None) -> dict:
    from sklearn.model_selection import train_test_split

    set_seed(seed)
    df = pd.read_csv(csv_path)
    label_col = label_col or next(
        (c for c in ("Class", "class", "label", "Label") if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"khong tim thay cot nhan trong {csv_path}; "
                         f"dung --label-col. Cac cot: {list(df.columns)[:15]}")

    raw = df[label_col]
    # CIC danh so 1..5 theo category; 5 = Benign trong ban 5-Cat.
    if raw.dtype.kind in "iuf":
        benign_code = raw.max()
        y = (raw != benign_code).astype(int).to_numpy()
        cats = raw.astype(str)
        log.info("Nhan so - coi ma lon nhat (%s) la Benign", benign_code)
    else:
        y = raw.astype(str).str.lower().ne(config.BENIGN_CATEGORY.lower()).astype(int).to_numpy()
        cats = raw.astype(str)

    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).fillna(0.0).to_numpy()
    log.info("CSV baseline: %d mau x %d feature (benign=%d)",
             X.shape[0], X.shape[1], int((y == 0).sum()))

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=cats, random_state=seed)

    results = {}
    for name in models:
        clf = fit_one(name, X[tr], y[tr], class_weight, seed)
        if clf is None:
            continue
        pred = clf.predict(X[te])
        results[name] = compute_metrics(y[te], pred, decision_scores(clf, X[te]),
                                        cats.iloc[te])
        log.info("csv/%-4s macroF1=%.4f  fpr_benign=%.4f", name,
                 results[name]["macro_f1"], results[name]["fpr_benign"])

    out = {
        **seed_stamp(),
        "csv_path": str(csv_path),
        "label_col": label_col,
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "note": ("Baseline doc lap tren CSV goc cua CIC. Khong ghep duoc theo "
                 "tung mau voi split Androguard (CSV khong co sha256) va khong "
                 "dung duoc cho phan obfuscation. Chi de doi chieu voi literature."),
        "results": results,
    }
    atomic_write_json(config.RESULTS_DIR / "csv_baseline.json", out)
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 - baseline ML")
    ap.add_argument("--features", type=Path, default=config.CLEAN_FEATURE_PATH)
    ap.add_argument("--models", nargs="*", default=list(MODEL_NAMES))
    ap.add_argument("--per-group-models", nargs="*", default=["rf"],
                    help="model dung cho Bang B; mac dinh chi rf cho nhanh")
    ap.add_argument("--groups", nargs="*", default=list(config.GROUPS_DEFAULT))
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--max-per-group", type=int, default=20000)
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--csv-baseline", type=Path, default=None,
                    help="duong dan CSV goc cua CIC de train baseline doi chieu")
    ap.add_argument("--label-col", default=None)
    args = ap.parse_args()

    config.ensure_dirs()
    cw = None if args.no_class_weight else "balanced"

    if args.csv_baseline:
        with timed(log, "baseline CSV cua CIC"):
            train_csv_baseline(args.csv_baseline, args.models, cw, args.seed,
                               args.label_col)
        return

    with timed(log, "Phase 3 - train baseline"):
        train_all(args.features, args.models, args.per_group_models,
                  args.min_df, args.max_per_group, args.groups, cw, args.seed)


if __name__ == "__main__":
    main()
