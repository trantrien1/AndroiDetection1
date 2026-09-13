"""Phase 5 - sinh ma tran ket qua (PLAN muc 7).

Contribution chinh khong phai mot con so tong, ma ba bang:

  Bang A  muc tut macro-F1 theo tung ky thuat + FPR benign + %APK hong
  Bang B  ma tran ky thuat x nhom feature - o nao sup thi phai thay bieu dien
  Bang C  kiem tra shortcut learning: feature space co ma hoa dau vet
          obfuscation khong

Module nay khong train lai gi cho Bang A/B - no doc ket qua da co tu
src.evaluate. Rieng Bang C phai train mot classifier phu.

Chay:
    python -m src.matrix
    python -m src.matrix --model rf --skip-table-c
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from . import config
from .features.vectorize import GroupVectorizer, load_feature_frame, parse_dicts
from .utils import atomic_write_json, read_json, seed_stamp, set_seed, setup_logging, timed

log = setup_logging("matrix")

NA = "—"


def _fmt_delta(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return NA
    return f"{v:+.4f}"


def _load_obf_results(techniques: list[str]) -> dict[str, dict]:
    out = {}
    for t in techniques:
        r = read_json(config.RESULTS_DIR / f"obf_{t}.json")
        if r:
            out[t] = r
        else:
            log.warning("[%s] chua co results/obf_%s.json - chay src.evaluate --obf", t, t)
    return out


# --------------------------------------------------------------------------
# Bang A
# --------------------------------------------------------------------------
def table_a(obf_results: dict[str, dict], model: str = "rf") -> pd.DataFrame:
    broken_csv = config.RESULTS_DIR / "obf_broken_rate.csv"
    broken = (pd.read_csv(broken_csv).set_index("technique")
              if broken_csv.exists() else pd.DataFrame())

    identical = _identical_feature_rate(list(obf_results))

    rows = []
    key = f"ALL__{model}"
    for tech in config.TECHNIQUE_ORDER:
        r = obf_results.get(tech)
        if not r:
            continue
        cln = r["metrics_clean_paired"].get(key, {})
        obf = r["metrics_obf"].get(key, {})
        f1c, f1o = cln.get("macro_f1"), obf.get("macro_f1")
        rows.append({
            "technique": tech,
            "obfuscators": "+".join(r.get("obfuscators", [])),
            "n_paired": r.get("n_paired", 0),
            "f1_clean": f1c,
            "f1_obf": f1o,
            "delta": (f1o - f1c) if (f1c is not None and f1o is not None) else None,
            "recall_malware_obf": obf.get("recall_malware"),
            "fpr_benign_obf": obf.get("fpr_benign"),
            "broken_pct": float(broken.loc[tech, "broken_pct"]) if tech in broken.index else None,
            "identical_features_pct": identical.get(tech),
        })
    return pd.DataFrame(rows)


def _identical_feature_rate(techniques: list[str]) -> dict[str, float]:
    """PLAN muc 8: kiem tra APK co THAT SU bi obfuscate khong.

    Neu vector feature cua ban obf trung y het ban sach thi ky thuat do khong
    cham vao bat ky thu gi ta do - va moi Delta ~ 0 o hang do la vo nghia chu
    khong phai bang chung ve do ben.
    """
    if not config.CLEAN_FEATURE_PATH.exists():
        return {}
    clean = load_feature_frame(config.CLEAN_FEATURE_PATH).set_index("sha256")
    out = {}
    for tech in techniques:
        p = config.technique_feature_path(tech)
        if not p.exists():
            continue
        obf = load_feature_frame(p).set_index("sha256")
        common = clean.index.intersection(obf.index)
        if common.empty:
            continue
        same = (clean.loc[common, "features_json"].to_numpy()
                == obf.loc[common, "features_json"].to_numpy())
        out[tech] = round(100.0 * float(same.mean()), 2)
    return out


# --------------------------------------------------------------------------
# Bang B
# --------------------------------------------------------------------------
def table_b(obf_results: dict[str, dict], model: str = "rf",
            groups: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Moi o la Delta F1 cua model chi-mot-nhom duoi mot ky thuat.

    Tra ve (bang delta, bang F1 sach theo nhom) - bang thu hai can de doc bang
    thu nhat: Delta = -0.02 tren mot nhom von chi dat F1 0.55 khi sach thi
    khong noi len dieu gi.
    """
    if groups is None:
        groups = []
        for r in obf_results.values():
            for k in r["metrics_obf"]:
                fs = k.split("__")[0]
                if fs != "ALL" and fs not in groups:
                    groups.append(fs)
        groups.sort()

    delta_rows, clean_rows = [], []
    for tech in config.TECHNIQUE_ORDER:
        r = obf_results.get(tech)
        if not r:
            continue
        d = {"technique": tech}
        c = {"technique": tech}
        for g in groups:
            key = f"{g}__{model}"
            cln = r["metrics_clean_paired"].get(key, {}).get("macro_f1")
            obf = r["metrics_obf"].get(key, {}).get("macro_f1")
            d[g] = (obf - cln) if (cln is not None and obf is not None) else None
            c[g] = cln
        delta_rows.append(d)
        clean_rows.append(c)
    return pd.DataFrame(delta_rows), pd.DataFrame(clean_rows)


# --------------------------------------------------------------------------
# Bang C - shortcut learning
# --------------------------------------------------------------------------
def table_c(techniques: list[str], n_splits: int = 5,
            seed: int = config.RANDOM_SEED) -> dict:
    """Train classifier chi de phan biet "da obfuscate / chua", bo qua nhan malware.

    Neu no dat F1 cao thi feature space dang ma hoa manh dau vet obfuscation,
    va moi ket qua detection deu dang nghi: model co the dang doc dau vet dong
    goi thay vi hanh vi.

    Chia fold theo sha256 (GroupKFold) - neu khong, ban sach va ban obf cua
    CUNG mot APK se nam ca o train lan test, va F1 se cao gia tao.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    set_seed(seed)
    vec = GroupVectorizer.load(config.MODELS_DIR / "vectorizer.joblib")
    clean = load_feature_frame(config.CLEAN_FEATURE_PATH)

    def _detector(X, y, groups) -> dict:
        # n_estimators=100: day la chan doan, khong phai baseline bao cao.
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                     random_state=seed, class_weight="balanced")
        n_g = len(np.unique(groups))
        k = min(n_splits, n_g)
        if k < 2:
            return {}
        pred = cross_val_predict(clf, X, y, cv=GroupKFold(n_splits=k), groups=groups)
        return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "n": int(len(y)), "n_obf": int((y == 1).sum())}

    per_tech: dict[str, dict] = {}
    obf_by_tech: dict[str, pd.DataFrame] = {}

    for tech in techniques:
        p = config.technique_feature_path(tech)
        if not p.exists():
            continue
        obf_df = load_feature_frame(p)
        cln_df = clean[clean["sha256"].isin(set(obf_df["sha256"]))].reset_index(drop=True)
        obf_df = obf_df[obf_df["sha256"].isin(set(cln_df["sha256"]))].reset_index(drop=True)
        if cln_df.empty:
            continue
        obf_by_tech[tech] = obf_df

        Xc = vec.transform(parse_dicts(cln_df))
        Xo = vec.transform(parse_dicts(obf_df))
        X = sparse.vstack([Xc, Xo]).tocsr()
        y = np.r_[np.zeros(Xc.shape[0], int), np.ones(Xo.shape[0], int)]
        g = np.r_[cln_df["sha256"].to_numpy(), obf_df["sha256"].to_numpy()]

        per_tech[tech] = _detector(X, y, g)
        log.info("[C] %-14s detector macro-F1 = %.4f", tech,
                 per_tech[tech].get("macro_f1", float("nan")))

    # --- gop tat ca ky thuat ------------------------------------------------
    # Moi APK dong gop DUNG mot ban sach va DUNG mot ban obf (ky thuat chon
    # ngau nhien, co seed). Neu gop ca 6 ban obf thi lop 1 dong hon lop 0 gap
    # sau lan, va RF se sup ve viec doan tat ca la "da obfuscate" - macro-F1
    # khi do bang nhau o moi nhom va khong con do duoc gi.
    per_group: dict[str, dict] = {}
    if obf_by_tech:
        rng = np.random.default_rng(seed)
        techs = sorted(obf_by_tech)
        sha_sets = {t: set(d["sha256"]) for t, d in obf_by_tech.items()}
        by_sha = {t: d.set_index("sha256") for t, d in obf_by_tech.items()}
        shas = sorted(set().union(*sha_sets.values()))

        picked_rows, picked_shas = [], []
        for sha in shas:
            avail = [t for t in techs if sha in sha_sets[t]]
            t = avail[int(rng.integers(len(avail)))]
            row = by_sha[t].loc[sha].copy()
            row["sha256"] = sha
            picked_rows.append(row)
            picked_shas.append(sha)

        obf_pool = pd.DataFrame(picked_rows)
        cln_pool = clean[clean["sha256"].isin(set(picked_shas))].reset_index(drop=True)
        obf_pool = obf_pool[obf_pool["sha256"].isin(set(cln_pool["sha256"]))].reset_index(drop=True)

        Xp = sparse.vstack([vec.transform(parse_dicts(cln_pool)),
                            vec.transform(parse_dicts(obf_pool))]).tocsr()
        yp = np.r_[np.zeros(len(cln_pool), int), np.ones(len(obf_pool), int)]
        gp = np.r_[cln_pool["sha256"].to_numpy(), obf_pool["sha256"].to_numpy()]

        per_group["ALL"] = _detector(Xp, yp, gp)
        for grp in vec.available_groups():
            per_group[grp] = _detector(vec.subset(Xp, [grp]), yp, gp)
        for grp, v in per_group.items():
            log.info("[C] pooled %-3s detector macro-F1 = %.4f", grp,
                     v.get("macro_f1", float("nan")))

    return {"per_technique": per_tech, "pooled_per_group": per_group,
            "pooled_note": ("Moi APK dong gop 1 ban sach + 1 ban obf (ky thuat "
                            "chon ngau nhien co seed) de hai lop can bang.")}


# --------------------------------------------------------------------------
# Canh bao tu dong (PLAN muc 8)
# --------------------------------------------------------------------------
def red_flags(a: pd.DataFrame, c: dict, trivial_threshold: float = 0.02,
              shortcut_threshold: float = 0.90) -> list[str]:
    flags: list[str] = []

    t1 = a[a["technique"] == "T1_trivial"]
    if not t1.empty and t1.iloc[0].get("delta") is not None:
        d = t1.iloc[0]["delta"]
        if d < -trivial_threshold:
            flags.append(
                f"T1 Trivial lam tut macro-F1 {d:+.4f}. Chi dong goi lai ma da tut "
                f"nghia la model dang hoc artifact dong goi, khong phai hanh vi. "
                f"Moi hang duoi phai duoc doc nhu 'them bao nhieu so voi T1', "
                f"khong phai so voi ban sach.")

    for _, row in a.iterrows():
        ident = row.get("identical_features_pct")
        if ident is not None and ident > 5:
            flags.append(
                f"{row['technique']}: {ident:.1f}% APK co vector feature TRUNG Y HET "
                f"ban sach. Ky thuat nay khong cham vao cai ta do - Delta o hang nay "
                f"khong phai bang chung ve do ben.")
        d = row.get("delta")
        # T1 khong tut la ket qua MONG MUON (model khong doc artifact dong goi),
        # nen khong canh bao o hang do - truong hop xau cua T1 da co flag rieng.
        if (d is not None and d > -0.005 and (ident or 0) <= 5
                and row["technique"] != "T1_trivial"):
            flags.append(
                f"{row['technique']}: Delta = {d:+.4f}, gan nhu khong tut. Kiem tra "
                f"Bang C va cot identical_features_pct truoc khi mung.")
        b = row.get("broken_pct")
        if b is not None and b > 30:
            flags.append(
                f"{row['technique']}: {b:.1f}% APK hong sau obfuscation. Tap con sau "
                f"khi loc co the da lech phan phoi - kiem tra con bao nhieu benign.")

    pooled = c.get("pooled_per_group", {}).get("ALL", {}).get("macro_f1")
    if pooled is not None and pooled >= shortcut_threshold:
        flags.append(
            f"Bang C: detector 'da obfuscate hay chua' dat macro-F1 {pooled:.4f}. "
            f"Feature space dang ma hoa manh dau vet obfuscation -> moi ket qua "
            f"detection o Bang A/B deu dang nghi.")
    for grp, v in sorted(c.get("pooled_per_group", {}).items()):
        if grp == "ALL":
            continue
        f1 = v.get("macro_f1")
        if f1 is not None and f1 >= shortcut_threshold:
            flags.append(f"Bang C: rieng nhom {grp} da du de nhan ra APK bi "
                         f"obfuscate (macro-F1 {f1:.4f}).")
    return flags


# --------------------------------------------------------------------------
# Ket xuat
# --------------------------------------------------------------------------
def _md(df: pd.DataFrame, fmt: dict[str, callable] | None = None) -> str:
    if df.empty:
        return "_(chua co du lieu)_\n"
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in fmt:
                cells.append(fmt[c](v))
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append(NA)
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def build_report(model: str = "rf", skip_c: bool = False,
                 techniques: list[str] | None = None) -> Path:
    techniques = techniques or config.TECHNIQUE_ORDER
    obf_results = _load_obf_results(techniques)
    if not obf_results:
        raise RuntimeError("khong co ket qua obfuscated nao - chay src.evaluate --obf truoc")

    a = table_a(obf_results, model)
    b_delta, b_clean = table_b(obf_results, model)

    c: dict = {}
    if not skip_c:
        with timed(log, "Bang C - detector shortcut learning"):
            c = table_c([t for t in techniques if config.technique_feature_path(t).exists()])

    flags = red_flags(a, c)

    clean_baseline = read_json(config.RESULTS_DIR / "clean_baseline.json", default={})
    csv_baseline = read_json(config.RESULTS_DIR / "csv_baseline.json", default={})

    def _pct(v):
        return NA if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}%"

    dfmt = {"delta": _fmt_delta, "broken_pct": _pct, "identical_features_pct": _pct}
    parts = [
        "# Ket qua - do ben cua static Android malware detection duoi obfuscation",
        "",
        f"Seed: `{config.RANDOM_SEED}` | model bao cao: `ALL__{model}` | "
        f"sinh luc: {seed_stamp()['generated_at']}",
        "",
        "## Baseline sach (Phase 3)", "",
    ]

    cb = clean_baseline.get("metrics", {})
    if cb:
        rows = [{"model": k, "n_features": NA, "macro_f1": v["macro_f1"],
                 "f1_benign": v["f1_benign"], "f1_malware": v["f1_malware"],
                 "fpr_benign": v["fpr_benign"]} for k, v in sorted(cb.items())]
        parts += [_md(pd.DataFrame(rows).drop(columns=["n_features"])), ""]
    if csv_baseline.get("results"):
        parts += ["### Doi chieu: CSV goc cua CIC", "",
                  "_" + csv_baseline.get("note", "") + "_", "",
                  _md(pd.DataFrame([{"model": k, "macro_f1": v["macro_f1"],
                                     "fpr_benign": v["fpr_benign"]}
                                    for k, v in csv_baseline["results"].items()])), ""]

    parts += [
        "## Bang A - muc tut macro-F1 theo ky thuat", "",
        "`f1_clean` duoc tinh lai tren dung nhung APK song sot qua ky thuat do "
        "(so sanh ghep cap), nen `delta` khong bi tron lan voi hieu ung loc mau.", "",
        _md(a, dfmt), "",
        "## Bang B - ma tran ky thuat x nhom feature", "",
        f"Moi o la Delta macro-F1 cua model chi-mot-nhom (`{model}`). "
        "Nhom nao bat bien thi giu, nhom nao sup thi thay bang bieu dien khac.", "",
        _md(b_delta, {g: _fmt_delta for g in b_delta.columns if g != "technique"}), "",
        "### Bang B-phu: F1 sach cua tung nhom (doc kem de khoi hieu nham Delta)", "",
        _md(b_clean), "",
        "## Bang C - kiem tra shortcut learning", "",
    ]

    if c:
        pt = pd.DataFrame([{"technique": k, **v} for k, v in c["per_technique"].items()])
        pg = pd.DataFrame([{"featureset": k, **v} for k, v in c["pooled_per_group"].items()])
        parts += ["Detector 'da obfuscate / chua', chia fold theo sha256.", "",
                  "**Theo tung ky thuat**", "", _md(pt), "",
                  "**Gop tat ca ky thuat, cat theo nhom feature**", "", _md(pg), ""]
    else:
        parts += ["_(bo qua bang C)_", ""]

    parts += ["## Canh bao tu dong", ""]
    parts += ([f"- {f}" for f in flags] if flags
              else ["- Khong co canh bao nao kich hoat."])
    parts += ["", "## Cach doc ba bang nay", "",
              "1. Doc Bang A theo hang T1 truoc. T1 la control: no do anh huong cua "
              "viec dong goi lai, khong phai cua obfuscation.",
              "2. Voi T2-T6, phan tut dang quan tam la phan VUOT QUA muc tut cua T1.",
              "3. Bang B chi ra phai sua gi. O nao am sau la nhom feature do bi ky "
              "thuat do pha; o nao gan 0 la nhom bat bien voi ky thuat do.",
              "4. Bang C phu quyet: neu detector obfuscation dat F1 cao, moi ket luan "
              "o Bang A/B deu dang nghi truoc khi dien giai.", ""]

    out_md = config.RESULTS_DIR / "tables.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(parts), encoding="utf-8")

    atomic_write_json(config.RESULTS_DIR / "tables.json", {
        **seed_stamp(), "model": model,
        "table_a": a.to_dict("records"),
        "table_b_delta": b_delta.to_dict("records"),
        "table_b_clean": b_clean.to_dict("records"),
        "table_c": c, "red_flags": flags,
    })
    a.to_csv(config.RESULTS_DIR / "table_a.csv", index=False)
    b_delta.to_csv(config.RESULTS_DIR / "table_b_delta.csv", index=False)

    log.info("-> %s", out_md)
    print("\n" + "\n".join(parts[:0] or []))
    print(_md(a, dfmt))
    print(_md(b_delta, {g: _fmt_delta for g in b_delta.columns if g != "technique"}))
    if flags:
        print("CANH BAO:")
        for f in flags:
            print("  - " + f)
    return out_md


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 - sinh ma tran ket qua")
    ap.add_argument("--model", default="rf", help="model dung cho Bang A/B")
    ap.add_argument("--techniques", nargs="*", default=None)
    ap.add_argument("--skip-table-c", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    build_report(args.model, args.skip_table_c, args.techniques)


if __name__ == "__main__":
    main()
