"""Chot split va chot tap con de obfuscate.

PLAN dat viec nay o Phase 3, nhung Phase 2 phai biet truoc split moi trich
dung 6.000 train + 500 test thay vi ca 17k. Vi vay module nay chay o cuoi
Phase 1, va tu do tro di test_sha256.txt KHONG BAO GIO duoc sinh lai.

PLAN muc 5: "Chot cung test_sha256.txt ngay tai day va khong bao gio doi."
Module nay tu choi ghi de file da ton tai tru khi co --force.

Chay:
    python -m src.split
    python -m src.split --n-train 6000 --n-val 1000 --n-test 500
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from . import config
from .utils import atomic_write_json, read_sha_list, seed_stamp, set_seed, setup_logging, write_sha_list

log = setup_logging("split")


def _stratified_take(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Lay n dong, giu ti le category cua df goc.

    Phan bo theo largest-remainder de tong dung bang n, va moi category co mat
    it nhat 1 mau neu n du lon.
    """
    if n >= len(df):
        return df
    counts = df["category"].value_counts()
    exact = {c: n * v / len(df) for c, v in counts.items()}
    take = {c: int(v) for c, v in exact.items()}
    # Bao dam category hiem khong bi mat han.
    for c in take:
        if take[c] == 0 and counts[c] > 0:
            take[c] = 1
    # Chia phan du theo phan le lon nhat.
    while sum(take.values()) > n:
        c = max(take, key=lambda k: (take[k] - exact[k], take[k]))
        take[c] -= 1
    order = sorted(exact, key=lambda c: exact[c] - int(exact[c]), reverse=True)
    i = 0
    while sum(take.values()) < n:
        c = order[i % len(order)]
        if take[c] < counts[c]:
            take[c] += 1
        i += 1

    parts = [g.sample(n=take[c], random_state=seed) for c, g in df.groupby("category") if take.get(c)]
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


def _balanced_take(df: pd.DataFrame, n_benign: int, n_malware: int,
                   seed: int) -> pd.DataFrame:
    """Lay n_benign benign + n_malware malware chia deu 4 category malware.

    Vi sao tap test lai can bang chu khong theo ti le tu nhien:

    PLAN muc 5 noi split stratified 60/20/20, con PLAN muc 6 doi tap de
    obfuscate la "500 APK stratified: 250 benign + 250 malware" lay tu
    test_sha256.txt. Hai cau do khong cung ton tai duoc: benign chi chiem
    ~23% dataset, nen mot tap test 500 theo ti le tu nhien chi co ~116 benign,
    khong bao gio du 250.

    Cach doc duy nhat nhat quan: 500 APK duoc trich feature o phia test CHINH
    LA 500 APK se bi obfuscate, can bang 250/250 - khop dung ngan sach Phase 2
    (500 sach + 500x6 obfuscated). Dat --test-stratified de quay ve ti le tu
    nhien; khi do tap obfuscate se khong con can bang.

    Tac dung phu dang ke: 250 benign cho ta mot uoc luong FPR on hon nhieu so
    voi 116, va FPR tren lop Benign la mot trong nhung so chinh cua Bang A.
    """
    benign = df[df["label"] == 0]
    n_b = min(n_benign, len(benign))
    if n_b < n_benign:
        log.warning("Chi co %d benign, can %d", len(benign), n_benign)
    picked = [benign.sample(n=n_b, random_state=seed)]

    mal_cats = [c for c in config.CATEGORIES if c != config.BENIGN_CATEGORY]
    pools = {c: df[df["category"] == c] for c in mal_cats}
    base, extra = divmod(n_malware, len(mal_cats))
    short = 0
    taken: dict[str, int] = {}
    for i, c in enumerate(mal_cats):
        want = base + (1 if i < extra else 0)
        taken[c] = min(want, len(pools[c]))
        short += want - taken[c]
    for c in mal_cats:                      # bu tu category con du mau
        if short <= 0:
            break
        add = min(short, len(pools[c]) - taken[c])
        taken[c] += add
        short -= add
    if short > 0:
        log.warning("Thieu %d malware so voi yeu cau %d", short, n_malware)

    for c in mal_cats:
        if taken[c]:
            picked.append(pools[c].sample(n=taken[c], random_state=seed))
    return pd.concat(picked).sample(frac=1.0, random_state=seed)


def make_splits(manifest_csv: Path = config.MANIFEST_CSV,
                n_train: int = config.N_TRAIN,
                n_val: int = config.N_VAL,
                n_test: int = config.N_TEST,
                seed: int = config.RANDOM_SEED,
                force: bool = False,
                test_balanced: bool = True) -> dict:
    set_seed(seed)
    df = pd.read_csv(manifest_csv, dtype={"sha256": str})
    log.info("manifest: %d APK | %s", len(df), dict(Counter(df["category"])))

    test_path = config.SPLIT_DIR / "test_sha256.txt"
    config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    if test_path.exists() and not force:
        # Tap test da chot -> chi sinh lai train/val tren phan con lai.
        frozen_test = set(read_sha_list(test_path))
        log.info("test_sha256.txt da ton tai (%d sha) - GIU NGUYEN", len(frozen_test))
        test = df[df["sha256"].isin(frozen_test)]
        if len(test) != len(frozen_test):
            log.warning("manifest thieu %d sha cua tap test da chot - kiem tra lai "
                        "xem APK co giai nen du khong",
                        len(frozen_test) - len(test))
        rest = df[~df["sha256"].isin(frozen_test)]
    else:
        if test_path.exists():
            log.warning("--force: GHI DE tap test da chot. Moi ket qua truoc do "
                        "khong con so sanh duoc voi ket qua sau day.")
        # Thanh phan can bang suy ra tu n_test, khong hardcode: n_test=500 ->
        # 250 benign + 250 malware, dung bang tap con obfuscate cua PLAN muc 6.
        test = (_balanced_take(df, n_test // 2, n_test - n_test // 2, seed)
                if test_balanced else _stratified_take(df, n_test, seed))
        rest = df[~df["sha256"].isin(set(test["sha256"]))]
        write_sha_list(test_path, test["sha256"].tolist())
        log.info("CHOT tap test: %d sha -> %s", len(test), test_path)

    train = _stratified_take(rest, n_train, seed)
    rest2 = rest[~rest["sha256"].isin(set(train["sha256"]))]
    val = _stratified_take(rest2, n_val, seed)

    for name, part in (("train", train), ("val", val), ("test", test)):
        write_sha_list(config.SPLIT_DIR / f"{name}_sha256.txt", part["sha256"].tolist())
        log.info("%-5s %5d | %s", name, len(part), dict(Counter(part["category"])))

    # Tap con de obfuscate (PLAN muc 6): 250 benign + 250 malware chia deu
    # 4 category malware, lay tu DUNG tap test da chot.
    obf = _obf_subsample(test, seed)
    write_sha_list(config.SPLIT_DIR / "obf_sha256.txt", obf["sha256"].tolist())
    log.info("obf   %5d | %s", len(obf), dict(Counter(obf["category"])))

    # Danh sach gop: day chinh la nhung APK duy nhat can trich feature.
    all_clean = pd.concat([train, val, test]).drop_duplicates(subset="sha256")
    write_sha_list(config.SPLIT_DIR / "extract_sha256.txt", all_clean["sha256"].tolist())

    summary = {
        **seed_stamp(),
        "manifest": str(manifest_csv),
        "n_manifest": len(df),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_obf_subsample": len(obf),
        "n_to_extract_clean": len(all_clean),
        "n_to_extract_obf": len(obf) * len(config.TECHNIQUES),
        "per_split_category": {
            "train": dict(Counter(train["category"])),
            "val": dict(Counter(val["category"])),
            "test": dict(Counter(test["category"])),
            "obf": dict(Counter(obf["category"])),
        },
        "test_frozen_at": str(test_path),
        "test_balanced": bool(test_balanced),
        "n_test_benign": int((test["label"] == 0).sum()),
    }
    atomic_write_json(config.SPLIT_DIR / "split_summary.json", summary)
    log.info("Tong APK phai trich feature: %d sach + %d obfuscated",
             summary["n_to_extract_clean"], summary["n_to_extract_obf"])
    return summary


def _obf_subsample(test: pd.DataFrame, seed: int) -> pd.DataFrame:
    """250 benign + 250 malware, chia deu 4 category malware (PLAN muc 6).

    Quy tac 2 cua PLAN: phai obfuscate CA benign lan malware, neu khong model
    se hoc luat tat "obfuscate = doc".
    """
    benign = test[test["label"] == 0]
    n_b = min(config.N_OBF_BENIGN, len(benign))
    picked = [benign.sample(n=n_b, random_state=seed)]

    mal_cats = [c for c in config.CATEGORIES if c != config.BENIGN_CATEGORY]
    per_cat = config.N_OBF_MALWARE // len(mal_cats)
    leftover = config.N_OBF_MALWARE - per_cat * len(mal_cats)
    pools: dict[str, pd.DataFrame] = {c: test[test["category"] == c] for c in mal_cats}

    taken: dict[str, int] = {}
    for c in mal_cats:
        k = min(per_cat, len(pools[c]))
        taken[c] = k
        leftover += per_cat - k
    # Bu phan thieu tu cac category con du mau.
    for c in mal_cats:
        if leftover <= 0:
            break
        extra = min(leftover, len(pools[c]) - taken[c])
        taken[c] += extra
        leftover -= extra
    if leftover > 0:
        log.warning("Tap test khong du malware de lay %d - thieu %d",
                    config.N_OBF_MALWARE, leftover)

    for c in mal_cats:
        if taken[c]:
            picked.append(pools[c].sample(n=taken[c], random_state=seed))
    return pd.concat(picked).sample(frac=1.0, random_state=seed)


def load_split(name: str) -> list[str]:
    return read_sha_list(config.SPLIT_DIR / f"{name}_sha256.txt")


def split_of() -> dict[str, str]:
    """Tra ve {sha256: 'train'|'val'|'test'} cho moi sha da chot."""
    out: dict[str, str] = {}
    for name in ("train", "val", "test"):
        p = config.SPLIT_DIR / f"{name}_sha256.txt"
        if p.exists():
            for sha in read_sha_list(p):
                out[sha] = name
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Chot split + tap con obfuscate")
    ap.add_argument("--manifest", type=Path, default=config.MANIFEST_CSV)
    ap.add_argument("--n-train", type=int, default=config.N_TRAIN)
    ap.add_argument("--n-val", type=int, default=config.N_VAL)
    ap.add_argument("--n-test", type=int, default=config.N_TEST,
                    help="mac dinh chia doi benign/malware; xem --test-stratified")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--force", action="store_true",
                    help="GHI DE test_sha256.txt da chot - pha vo moi so sanh cu")
    ap.add_argument("--test-stratified", action="store_true",
                    help="tap test theo ti le tu nhien thay vi can bang 250/250; "
                         "khi do tap obfuscate se khong con 250 benign")
    args = ap.parse_args()

    config.ensure_dirs()
    make_splits(args.manifest, args.n_train, args.n_val, args.n_test,
                args.seed, args.force, not args.test_stratified)


if __name__ == "__main__":
    main()
