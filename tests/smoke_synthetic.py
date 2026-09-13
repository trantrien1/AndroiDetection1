"""Smoke test duong ong ML tren du lieu tong hop - khong can APK, khong can Colab.

Muc dich: kiem tra split -> vectorize -> train -> evaluate -> matrix chay dung
TRUOC khi bo 10 gio vao Phase 4. Neu test nay hong thi khong co ly do gi de
chay obfuscation.

Test sinh feature gia co tin hieu dat vao dung nhung cho ma PLAN du doan:
G4 (string) mang tin hieu manh nhat, va T3 String encrypt pha huy no. Ket qua
mong doi la o [T3, G4] cua Bang B am sau, cac o con lai gan 0.

Chay:
    python tests/smoke_synthetic.py
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TMP = Path(tempfile.mkdtemp(prefix="apkrob_smoke_"))
os.environ["APKROB_WORK"] = str(TMP / "work")
os.environ["APKROB_SCRATCH"] = str(TMP / "scratch")

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from src import config                                      # noqa: E402
from src.features import groups as G                        # noqa: E402

N_PER = {"Benign": 400, "Adware": 200, "Banking": 200, "SMS": 200, "Riskware": 200}
PERMS = [f"android.permission.P{i}" for i in range(40)]
APIS = [f"Landroid/x/C{i};->m{i}" for i in range(300)]
OPS = ["invoke-virtual", "move-result", "const-string", "goto", "nop", "if-eqz",
       "iget-object", "return-void", "new-instance", "aput"]
DOMS = [f"d{i}.example.com" for i in range(60)]


def make_dict(cat: str, seed: int) -> dict[str, float]:
    r = random.Random(seed)
    mal = cat != "Benign"
    f: dict[str, float] = {}

    for p in r.sample(PERMS, r.randint(5, 15)):
        f[G.key("G1", f"perm:{p}")] = 1.0
    if mal and r.random() < 0.75:
        f[G.key("G1", "perm:android.permission.P0")] = 1.0
    f[G.key("G1", "n_activity")] = float(r.randint(1, 30))
    f[G.key("G1", "n_service")] = float(r.randint(0, 8) + (3 if mal else 0))
    f[G.key("G1", "min_sdk")] = float(r.choice([14, 15, 16, 19, 21]))

    for a in r.sample(APIS, r.randint(20, 60)):
        f[G.key("G2", f"api:{a}")] = float(r.randint(1, 20))
    if mal and r.random() < 0.8:
        f[G.key("G2", "sens:sendTextMessage")] = float(r.randint(1, 5))
    f[G.key("G2", "n_distinct_api")] = float(r.randint(20, 80))

    n_ins = r.randint(5000, 50000)
    f[G.key("G3", "n_instructions")] = float(n_ins)
    f[G.key("G3", "n_methods")] = float(r.randint(200, 3000))
    rem = n_ins
    for op in OPS:
        c = r.randint(1, max(2, rem // 8))
        f[G.key("G3", f"op:{op}")] = float(c)
        rem -= c
    for _ in range(15):
        f[G.key("G3", f"op2:{r.choice(OPS)}|{r.choice(OPS)}")] = float(r.randint(1, 500))

    n_dom = r.randint(1, 6) + (4 if mal else 0)
    for d in r.sample(DOMS, min(n_dom, len(DOMS))):
        f[G.key("G4", f"dom:{d}")] = float(r.randint(1, 3))
    if mal and r.random() < 0.85:
        f[G.key("G4", "dom:d0.example.com")] = 1.0
        f[G.key("G4", "kw:sms")] = float(r.randint(1, 9))
    f[G.key("G4", "n_strings")] = float(r.randint(300, 5000))
    f[G.key("G4", "n_urls")] = float(n_dom)
    f[G.key("G4", "entropy_mean")] = 3.0 + r.random()
    f[G.key("G4", "ratio_high_entropy")] = r.random() * 0.05
    return f


def perturb(d: dict[str, float], tech: str, r: random.Random) -> dict[str, float]:
    """Mo phong tac dong cua tung ky thuat len feature space."""
    o = dict(d)
    if tech == "T1_trivial":
        for k in list(o):
            if k.startswith("G3:op:") and r.random() < 0.3:
                o[k] *= 1 + r.random() * 0.02
    elif tech == "T2_rename":
        for k in list(o):
            if k.startswith("G2:api:") and r.random() < 0.6:
                o.pop(k)
                o[f"G2:api:La{r.randint(0, 99999)};->m"] = 1.0
    elif tech == "T3_string":
        for k in list(o):
            if k.startswith(("G4:dom:", "G4:kw:")):
                o.pop(k)
        o["G4:n_urls"] = 0.0
        o["G4:entropy_mean"] = 5.4
        o["G4:ratio_high_entropy"] = 0.6
    elif tech == "T4_asset":
        o["G4:entropy_mean"] = o.get("G4:entropy_mean", 3.0) + 1.0
        o["G1:n_assets"] = 3.0
    elif tech == "T5_cfg":
        n = o.get("G3:n_instructions", 0.0)
        o["G3:op:goto"] = o.get("G3:op:goto", 0.0) + n * 0.15
        o["G3:op:nop"] = o.get("G3:op:nop", 0.0) + n * 0.10
        o["G3:n_instructions"] = n * 1.3
    elif tech == "T6_reflection":
        for k in list(o):
            if k.startswith("G2:api:") and r.random() < 0.7:
                o.pop(k)
        o["G2:sens:invoke"] = 50.0
    return o


def _to_parquet(records: dict[str, dict], man: pd.DataFrame, path: Path, source: str) -> None:
    df = pd.DataFrame([{"sha256": sha, "n_features": len(d), "extract_s": 1.0,
                        "features_json": json.dumps(d, separators=(",", ":"))}
                       for sha, d in records.items()])
    df = df.merge(man[["sha256", "category", "label"]], on="sha256")
    df["source"] = source
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def main() -> int:
    config.ensure_dirs()

    cats = [c for c, n in N_PER.items() for _ in range(n)]
    man = pd.DataFrame([{"sha256": f"{i:064x}", "category": c,
                         "label": config.category_to_label(c),
                         "path": f"/fake/{c}/{i}.apk", "size": 1000 + i}
                        for i, c in enumerate(cats)])
    man.to_csv(config.MANIFEST_CSV, index=False)

    from src.split import make_splits
    make_splits(config.MANIFEST_CSV, n_train=700, n_val=150, n_test=200)

    feats = {r.sha256: make_dict(r.category, i) for i, r in enumerate(man.itertuples())}
    _to_parquet(feats, man, config.CLEAN_FEATURE_PATH, "clean")

    from src.utils import read_sha_list
    obf_shas = read_sha_list(config.SPLIT_DIR / "obf_sha256.txt")

    broken_rows = []
    for tech in config.TECHNIQUE_ORDER:
        r = random.Random(abs(hash(tech)) % 1000)
        keep = [s for s in obf_shas if r.random() > 0.08]     # gia lap 8% hong
        recs = {s: perturb(feats[s], tech, r) for s in keep}
        _to_parquet(recs, man, config.technique_feature_path(tech), tech)
        broken_rows.append({
            "technique": tech, "obfuscators": "+".join(config.TECHNIQUES[tech]),
            "n_attempted": len(obf_shas), "n_ok": len(keep),
            "n_broken": len(obf_shas) - len(keep),
            "broken_pct": round(100 * (len(obf_shas) - len(keep)) / len(obf_shas), 2),
            "median_s": 40.0, "top_reasons": "gia lap"})
    pd.DataFrame(broken_rows).to_csv(config.RESULTS_DIR / "obf_broken_rate.csv", index=False)

    from src.train import train_all
    train_all(config.CLEAN_FEATURE_PATH, ["rf", "svm"], ["rf"], 3, 20000,
              list(config.GROUPS_DEFAULT), "balanced", config.RANDOM_SEED)

    from src.evaluate import evaluate_clean, evaluate_obfuscated
    evaluate_clean("test")
    evaluate_obfuscated(list(config.TECHNIQUE_ORDER))

    from src.matrix import build_report
    build_report("rf", skip_c=False)

    # --- kiem tra ket qua co dung hinh dang mong doi khong -----------------
    tables = json.loads((config.RESULTS_DIR / "tables.json").read_text(encoding="utf-8"))
    b = {r["technique"]: r for r in tables["table_b_delta"]}
    fails = []

    d_t3_g4 = b.get("T3_string", {}).get("G4")
    if d_t3_g4 is None or d_t3_g4 > -0.3:
        fails.append(f"[T3, G4] phai sup sau (ky vong < -0.3), thuc te {d_t3_g4}")

    d_t3_g1 = b.get("T3_string", {}).get("G1")
    if d_t3_g1 is None or abs(d_t3_g1) > 0.05:
        fails.append(f"[T3, G1] phai gan 0 (T3 khong cham manifest), thuc te {d_t3_g1}")

    pooled = tables["table_c"]["pooled_per_group"]
    vals = [v["macro_f1"] for v in pooled.values() if "macro_f1" in v]
    if len(set(round(v, 6) for v in vals)) == 1:
        fails.append(f"Bang C gop suy bien: moi nhom cho cung mot so {vals[0]:.4f}")

    a = {r["technique"]: r for r in tables["table_a"]}
    if a.get("T3_string", {}).get("delta", 0) > -0.05:
        fails.append("Bang A: T3 phai lam tut macro-F1 tren feature set day du")

    print("\n" + "=" * 70)
    if fails:
        for f in fails:
            print("FAIL:", f)
        print("=" * 70)
        return 1
    print("SMOKE TEST PASS - duong ong ML chay dung tu split den Bang A/B/C")
    print("Ket qua tam:", config.RESULTS_DIR / "tables.md")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    code = main()
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(code)
