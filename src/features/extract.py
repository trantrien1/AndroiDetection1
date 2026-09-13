"""Phase 2 - trich feature tinh bang Androguard (PLAN muc 4).

Day la dieu kien can de thi nghiem ton tai, khong phai buoc toi uu: APK da
obfuscate khong co dong CSV tuong ung trong bo cua CIC nen buoc phai tu trich.

Ba rang buoc thiet ke:
  1. Timeout 60s/APK - Androguard treo han tren APK hong.
  2. Checkpoint moi 500 APK xuong Drive - Colab chet bat cu luc nao.
  3. Ghi log rieng APK fail kem ly do - ti le fail la mot ket qua phai bao cao.

Chay:
    # APK sach
    python -m src.features.extract --tag clean

    # APK da obfuscate cua mot ky thuat
    python -m src.features.extract --tag T3_string
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pandas as pd

from .. import config
from ..utils import (atomic_write_json, chunked, read_sha_list, seed_stamp,
                     setup_logging, sha256_file, shannon_entropy, timed)
from . import groups as G

log = setup_logging("extract")

# Bat trong worker qua initializer.
_WITH_G5 = False
_TIMEOUT = config.EXTRACT_TIMEOUT_S


# --------------------------------------------------------------------------
# Import Androguard - chiu ca 3.x lan 4.x
# --------------------------------------------------------------------------
def _androguard():
    try:
        from androguard.core.apk import APK
        from androguard.core.dex import DEX
        return APK, DEX
    except ImportError:
        from androguard.core.bytecodes.apk import APK              # type: ignore
        from androguard.core.bytecodes.dvm import DalvikVMFormat as DEX  # type: ignore
        return APK, DEX


def _as_str(s) -> str:
    if isinstance(s, str):
        return s
    for attr in ("get", "get_value"):
        f = getattr(s, attr, None)
        if callable(f):
            try:
                v = f()
                if isinstance(v, str):
                    return v
            except Exception:
                pass
    if isinstance(s, (bytes, bytearray)):
        return s.decode("utf-8", errors="replace")
    return str(s)


# --------------------------------------------------------------------------
# G1 - Manifest
# --------------------------------------------------------------------------
def extract_g1(a) -> dict[str, float]:
    f: dict[str, float] = {}

    for p in a.get_permissions() or []:
        f[G.key(G.G1, f"perm:{p}")] = 1.0
    try:
        for p in a.get_declared_permissions() or []:
            f[G.key(G.G1, f"declperm:{p}")] = 1.0
    except Exception:
        pass
    try:
        for feat in (a.get_features() or []):
            f[G.key(G.G1, f"uses-feature:{feat}")] = 1.0
    except Exception:
        pass

    counts = {"activity": 0, "service": 0, "receiver": 0, "provider": 0}
    exported = {k: 0 for k in counts}
    n_intent_filter = 0

    try:
        axml = a.get_android_manifest_xml()
    except Exception:
        axml = None

    if axml is not None:
        for tag in G.COMPONENT_TAGS:
            base = "activity" if tag == "activity-alias" else tag
            for el in axml.iter(tag):
                counts[base] = counts.get(base, 0) + 1
                if (el.get(G.ANDROID_NS + "exported") or "").lower() == "true":
                    exported[base] = exported.get(base, 0) + 1
        for el in axml.iter("intent-filter"):
            n_intent_filter += 1
            for act in el.iter("action"):
                name = act.get(G.ANDROID_NS + "name")
                if name:
                    f[G.key(G.G1, f"intent:{name}")] = 1.0
            for cat in el.iter("category"):
                name = cat.get(G.ANDROID_NS + "name")
                if name:
                    f[G.key(G.G1, f"category:{name}")] = 1.0

    for k, v in counts.items():
        f[G.key(G.G1, f"n_{k}")] = float(v)
    for k, v in exported.items():
        f[G.key(G.G1, f"n_exported_{k}")] = float(v)
    f[G.key(G.G1, "n_intent_filter")] = float(n_intent_filter)
    f[G.key(G.G1, "n_permissions")] = float(len(a.get_permissions() or []))

    def _sdk(method_name: str, default: float = 0.0) -> float:
        # Tra ve theo TEN: get_max_sdk_version() khong co o moi ban Androguard,
        # va truy cap thuoc tinh truc tiep se nem AttributeError ngoai pham vi try.
        fn = getattr(a, method_name, None)
        if fn is None:
            return default
        try:
            v = fn()
            return float(v) if v not in (None, "") else default
        except (TypeError, ValueError, AttributeError):
            return default

    f[G.key(G.G1, "min_sdk")] = _sdk("get_min_sdk_version")
    f[G.key(G.G1, "target_sdk")] = _sdk("get_target_sdk_version")
    f[G.key(G.G1, "max_sdk")] = _sdk("get_max_sdk_version")

    try:
        files = a.get_files() or []
        f[G.key(G.G1, "n_files")] = float(len(files))
        f[G.key(G.G1, "n_native_libs")] = float(sum(1 for x in files if x.endswith(".so")))
        f[G.key(G.G1, "n_dex")] = float(sum(1 for x in files if x.endswith(".dex")))
        f[G.key(G.G1, "n_assets")] = float(sum(1 for x in files if x.startswith("assets/")))
    except Exception:
        pass

    return f


# --------------------------------------------------------------------------
# G2 + G3 + G4 - mot lan duyet DEX duy nhat
#
# Gop ba nhom vao mot vong lap vi buoc dat nhat la giai ma instruction; duyet
# ba lan se lam Phase 2 dat gap ba.
# --------------------------------------------------------------------------
def extract_dex_groups(dex_list) -> dict[str, float]:
    api_counts: Counter[str] = Counter()
    api_classes: Counter[str] = Counter()
    op_counts: Counter[str] = Counter()
    op2_counts: Counter[str] = Counter()
    n_methods = 0
    n_instructions = 0
    all_strings: list[str] = []

    for dex in dex_list:
        try:
            methods = dex.get_methods()
        except Exception:
            continue
        for m in methods:
            n_methods += 1
            try:
                instrs = m.get_instructions()
            except Exception:
                continue
            prev = None
            for ins in instrs:
                try:
                    name = ins.get_name()
                except Exception:
                    continue
                n_instructions += 1
                op_counts[name] += 1
                if prev is not None:
                    op2_counts[f"{prev}|{name}"] += 1   # 2-gram trong pham vi method
                prev = name

                if name.startswith("invoke"):
                    try:
                        out = ins.get_output()
                    except Exception:
                        continue
                    mt = G.INVOKE_TARGET_RE.search(out or "")
                    if not mt:
                        continue
                    clazz, meth = mt.group(1), mt.group(2)
                    if G.is_framework_class(clazz):
                        api_counts[f"{clazz}->{meth}"] += 1
                        api_classes[clazz] += 1

        try:
            all_strings.extend(_as_str(s) for s in dex.get_strings())
        except Exception:
            pass

    f: dict[str, float] = {}

    # --- G2 -------------------------------------------------------------
    for name, c in api_counts.most_common(G.MAX_API_TOKENS):
        f[G.key(G.G2, f"api:{name}")] = float(c)
    for name, c in api_classes.most_common(G.MAX_API_TOKENS // 4):
        f[G.key(G.G2, f"cls:{name}")] = float(c)
    f[G.key(G.G2, "n_distinct_api")] = float(len(api_counts))
    f[G.key(G.G2, "n_api_calls")] = float(sum(api_counts.values()))
    for hint in G.SENSITIVE_API_HINTS:
        hits = sum(c for k, c in api_counts.items() if k.endswith("->" + hint))
        if hits:
            f[G.key(G.G2, f"sens:{hint}")] = float(hits)

    # --- G3 -------------------------------------------------------------
    # Luu count tho; vectorize.py se L1-normalize nhom G3 thanh phan bo.
    for name, c in op_counts.items():
        f[G.key(G.G3, f"op:{name}")] = float(c)
    for name, c in op2_counts.most_common(G.MAX_OPCODE_2GRAMS):
        f[G.key(G.G3, f"op2:{name}")] = float(c)
    f[G.key(G.G3, "n_methods")] = float(n_methods)
    f[G.key(G.G3, "n_instructions")] = float(n_instructions)
    f[G.key(G.G3, "avg_method_len")] = float(n_instructions / n_methods) if n_methods else 0.0
    f[G.key(G.G3, "n_distinct_opcode")] = float(len(op_counts))

    # --- G4 -------------------------------------------------------------
    f.update(extract_g4(all_strings))
    return f


def extract_g4(strings: list[str]) -> dict[str, float]:
    """URL, IP, so dien thoai, chuoi base64, entropy (PLAN muc 4).

    Du doan cua PLAN: day la nhom nhay nhat voi T3 String encrypt. Neu dung,
    entropy_mean va n_high_entropy phai tang vot con n_urls phai ve gan 0.
    """
    f: dict[str, float] = {}
    n_url = n_ip = n_phone = n_b64 = n_hex = n_high = 0
    ent_sum = 0.0
    ent_max = 0.0
    ent_n = 0
    len_sum = 0
    domains: Counter[str] = Counter()
    kw_hits: Counter[str] = Counter()

    for s in strings:
        if not s:
            continue
        len_sum += len(s)

        if G.URL_RE.search(s):
            n_url += 1
            for d in G.DOMAIN_RE.findall(s):
                domains[d.lower()] += 1
        if G.IP_RE.search(s):
            n_ip += 1
        if len(s) >= 9 and G.PHONE_RE.search(s):
            n_phone += 1
        if G.BASE64_RE.match(s):
            n_b64 += 1
        if G.HEXBLOB_RE.match(s):
            n_hex += 1

        if len(s) >= G.MIN_ENTROPY_LEN:
            e = shannon_entropy(s)
            ent_sum += e
            ent_n += 1
            ent_max = max(ent_max, e)
            if e >= G.HIGH_ENTROPY_THRESHOLD:
                n_high += 1

        low = s.lower()
        for kw in G.STRING_KEYWORDS:
            if kw in low:
                kw_hits[kw] += 1

    n = len(strings)
    f[G.key(G.G4, "n_strings")] = float(n)
    f[G.key(G.G4, "n_urls")] = float(n_url)
    f[G.key(G.G4, "n_ips")] = float(n_ip)
    f[G.key(G.G4, "n_phones")] = float(n_phone)
    f[G.key(G.G4, "n_base64")] = float(n_b64)
    f[G.key(G.G4, "n_hexblob")] = float(n_hex)
    f[G.key(G.G4, "n_high_entropy")] = float(n_high)
    f[G.key(G.G4, "n_distinct_domains")] = float(len(domains))
    f[G.key(G.G4, "entropy_mean")] = float(ent_sum / ent_n) if ent_n else 0.0
    f[G.key(G.G4, "entropy_max")] = float(ent_max)
    f[G.key(G.G4, "avg_str_len")] = float(len_sum / n) if n else 0.0
    # Ti le - ben hon count tho khi so hai APK khac kich thuoc.
    f[G.key(G.G4, "ratio_high_entropy")] = float(n_high / n) if n else 0.0
    f[G.key(G.G4, "ratio_base64")] = float(n_b64 / n) if n else 0.0

    for d, c in domains.most_common(G.MAX_DOMAIN_TOKENS):
        f[G.key(G.G4, f"dom:{d}")] = float(c)
    for kw, c in kw_hits.items():
        f[G.key(G.G4, f"kw:{kw}")] = float(c)
    return f


# --------------------------------------------------------------------------
# G5 - call graph (opt-in, PLAN muc 4 hoan sang pass sau)
# --------------------------------------------------------------------------
def extract_g5(apk_path: str) -> dict[str, float]:
    """Chi goi khi --with-g5. get_call_graph() ton 5-30s/APK, tuc ~80% tong
    thoi gian Phase 2 neu bat tu dau. Chi bat neu Bang B cho thay G1-G4 khong
    du tach bach.
    """
    from androguard.misc import AnalyzeAPK
    _, _, dx = AnalyzeAPK(apk_path)
    cg = dx.get_call_graph()

    n_nodes = cg.number_of_nodes()
    n_edges = cg.number_of_edges()
    f = {
        G.key(G.G5, "n_nodes"): float(n_nodes),
        G.key(G.G5, "n_edges"): float(n_edges),
        G.key(G.G5, "density"): float(n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0,
        G.key(G.G5, "avg_degree"): float(2 * n_edges / n_nodes) if n_nodes else 0.0,
    }
    try:
        import networkx as nx
        degs = [d for _, d in cg.degree()]
        f[G.key(G.G5, "max_degree")] = float(max(degs)) if degs else 0.0
        scc = list(nx.strongly_connected_components(cg))
        f[G.key(G.G5, "n_scc")] = float(len(scc))
        f[G.key(G.G5, "max_scc")] = float(max((len(c) for c in scc), default=0))
        f[G.key(G.G5, "n_leaf")] = float(sum(1 for _, d in cg.out_degree() if d == 0))
        f[G.key(G.G5, "n_root")] = float(sum(1 for _, d in cg.in_degree() if d == 0))
        # Histogram bac - hinh dang do thi, bat bien voi viec doi ten.
        for lo, hi in ((1, 1), (2, 3), (4, 7), (8, 15), (16, 10 ** 9)):
            c = sum(1 for d in degs if lo <= d <= hi)
            f[G.key(G.G5, f"deg_{lo}_{hi if hi < 10 ** 9 else 'inf'}")] = float(c)
    except ImportError:
        pass
    return f


# --------------------------------------------------------------------------
# Trich mot APK
# --------------------------------------------------------------------------
class ExtractTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise ExtractTimeout("het gio")


def extract_apk(apk_path: str, with_g5: bool = False) -> dict[str, float]:
    APK, DEX = _androguard()
    a = APK(apk_path)
    if not a.is_valid_APK():
        raise ValueError("APK khong hop le")

    f = extract_g1(a)

    dex_list = []
    for raw in a.get_all_dex():
        try:
            dex_list.append(DEX(raw))
        except Exception:
            continue
    if not dex_list:
        raise ValueError("khong doc duoc dex nao")

    f.update(extract_dex_groups(dex_list))
    if with_g5:
        f.update(extract_g5(apk_path))
    return f


def _init_worker(with_g5: bool, timeout: int) -> None:
    global _WITH_G5, _TIMEOUT
    _WITH_G5, _TIMEOUT = with_g5, timeout


def _work(task: tuple[str, str]) -> dict:
    """Chay trong worker. (sha256, apk_path) -> ban ghi ket qua.

    SIGALRM la cach duy nhat cat duoc mot APK treo ma khong giet ca pool.
    Windows khong co SIGALRM; o do buoc nay chay khong timeout - chap nhan
    duoc vi pipeline that su chay tren Colab (Linux).
    """
    sha, path = task
    t0 = time.time()
    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(_TIMEOUT)
    try:
        feats = extract_apk(path, _WITH_G5)
        return {"sha256": sha, "ok": True, "features": feats,
                "n_features": len(feats), "extract_s": round(time.time() - t0, 2),
                "error": ""}
    except ExtractTimeout:
        return {"sha256": sha, "ok": False, "features": {}, "n_features": 0,
                "extract_s": round(time.time() - t0, 2),
                "error": f"timeout>{_TIMEOUT}s"}
    except Exception as e:
        return {"sha256": sha, "ok": False, "features": {}, "n_features": 0,
                "extract_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {e}"[:300]}
    finally:
        if has_alarm:
            signal.alarm(0)


# --------------------------------------------------------------------------
# Driver co checkpoint + resume
# --------------------------------------------------------------------------
def parts_dir(tag: str) -> Path:
    return config.FEATURES_DIR / "parts" / tag


def already_done(tag: str) -> set[str]:
    d = parts_dir(tag)
    if not d.is_dir():
        return set()
    done: set[str] = set()
    for p in sorted(d.glob("features_part_*.parquet")):
        try:
            done |= set(pd.read_parquet(p, columns=["sha256"])["sha256"])
        except Exception as e:
            log.warning("part hong, bo qua: %s (%s)", p.name, e)
    return done


def run_extraction(tasks: list[tuple[str, str]], tag: str, workers: int,
                   with_g5: bool, timeout: int, chunk: int,
                   meta: pd.DataFrame | None = None) -> Path:
    d = parts_dir(tag)
    d.mkdir(parents=True, exist_ok=True)

    done = already_done(tag)
    if done:
        log.info("[%s] resume: %d APK da trich truoc do", tag, len(done))
    pending = [t for t in tasks if t[0] not in done]
    log.info("[%s] con lai %d/%d APK", tag, len(pending), len(tasks))
    if not pending:
        return merge_parts(tag, meta)

    failed_csv = config.FEATURES_DIR / f"failed_{tag}.csv"
    new_fail = not failed_csv.exists()
    fh = open(failed_csv, "a", newline="", encoding="utf-8")
    fw = csv.writer(fh)
    if new_fail:
        fw.writerow(["sha256", "path", "error", "extract_s"])

    existing_parts = len(list(d.glob("features_part_*.parquet")))
    t_start = time.time()
    n_ok = n_fail = 0

    try:
        # Mot pool MOI cho moi lo. Neu mot worker chet han (OOM, segfault trong
        # thu vien C), BrokenProcessPool chi lam mat mot lo chu khong giet ca
        # lan chay - va lo do se duoc lam lai o lan resume sau.
        for i, batch in enumerate(chunked(pending, chunk)):
            rows = []
            try:
                with ProcessPoolExecutor(max_workers=workers,
                                         initializer=_init_worker,
                                         initargs=(with_g5, timeout)) as ex:
                    for res, (sha, path) in zip(ex.map(_work, batch), batch):
                        if res["ok"]:
                            n_ok += 1
                            rows.append({
                                "sha256": res["sha256"],
                                "n_features": res["n_features"],
                                "extract_s": res["extract_s"],
                                "features_json": json.dumps(res["features"],
                                                            separators=(",", ":")),
                            })
                        else:
                            n_fail += 1
                            fw.writerow([sha, path, res["error"], res["extract_s"]])
            except BrokenProcessPool as e:
                log.error("[%s] pool chet o lo %d (%s) - giu lai %d ket qua, "
                          "phan con lai se lam o lan chay sau", tag, i, e, len(rows))
            fh.flush()

            if rows:
                part = d / f"features_part_{existing_parts + i:04d}.parquet"
                tmp = part.with_suffix(".parquet.tmp")
                pd.DataFrame(rows).to_parquet(tmp, index=False, compression="zstd")
                os.replace(tmp, part)

            elapsed = time.time() - t_start
            seen = n_ok + n_fail
            rate = seen / elapsed if elapsed else 0
            eta = (len(pending) - seen) / rate if rate else 0
            log.info("[%s] %d/%d ok=%d fail=%d | %.1f APK/s | ETA %.0f phut",
                     tag, seen, len(pending), n_ok, n_fail, rate, eta / 60)
    finally:
        fh.close()

    if n_fail:
        log.warning("[%s] %d APK fail (%.1f%%) -> %s",
                    tag, n_fail, 100 * n_fail / max(1, n_ok + n_fail), failed_csv)
    return merge_parts(tag, meta)


def merge_parts(tag: str, meta: pd.DataFrame | None = None) -> Path:
    """Gop cac part thanh mot parquet duy nhat, index theo sha256."""
    d = parts_dir(tag)
    parts = sorted(d.glob("features_part_*.parquet"))
    if not parts:
        raise RuntimeError(f"[{tag}] khong co part nao de gop")

    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset="sha256", keep="last")

    if meta is not None:
        df = df.merge(meta[["sha256", "category", "label"]], on="sha256", how="left")
        if df["label"].isna().any():
            n = int(df["label"].isna().sum())
            log.warning("[%s] %d dong khong khop duoc nhan - bo", tag, n)
            df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

    df["source"] = tag
    out = (config.CLEAN_FEATURE_PATH if tag == "clean"
           else config.technique_feature_path(tag))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, out)

    log.info("[%s] gop %d part -> %d APK -> %s (%.1f MB)", tag, len(parts), len(df),
             out, out.stat().st_size / 1e6)

    stats = {
        **seed_stamp(), "tag": tag, "n_apk": len(df),
        "median_features_per_apk": float(df["n_features"].median()),
        "median_extract_s": float(df["extract_s"].median()),
        "total_extract_s": float(df["extract_s"].sum()),
    }
    failed_csv = config.FEATURES_DIR / f"failed_{tag}.csv"
    if failed_csv.exists():
        fdf = pd.read_csv(failed_csv).drop_duplicates(subset="sha256")
        stats["n_failed"] = len(fdf)
        stats["fail_rate"] = len(fdf) / max(1, len(fdf) + len(df))
        stats["fail_reasons"] = dict(Counter(
            fdf["error"].astype(str).str.split(":").str[0]).most_common(10))
    atomic_write_json(config.FEATURES_DIR / f"extract_stats_{tag}.json", stats)
    return out


# --------------------------------------------------------------------------
def build_tasks(tag: str, manifest: pd.DataFrame, sha_list: list[str] | None,
                apk_dir: Path | None) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """Tra ve [(sha256_goc, duong_dan_apk)] + metadata nhan.

    Voi APK da obfuscate, ten file la <sha256_goc>.apk. Giu sha256 goc lam khoa
    de noi duoc voi nhan va voi ban sach tuong ung.
    """
    if tag == "clean":
        sub = manifest[manifest["sha256"].isin(sha_list)] if sha_list else manifest
        missing = set(sha_list or []) - set(sub["sha256"])
        if missing:
            log.warning("%d sha trong danh sach khong co trong manifest", len(missing))
        tasks = list(zip(sub["sha256"], sub["path"]))
    else:
        apk_dir = apk_dir or (config.APK_OBF_DIR / tag)
        if not apk_dir.is_dir():
            raise FileNotFoundError(f"khong tim thay thu muc APK obfuscated: {apk_dir}")
        tasks = [(p.stem, str(p)) for p in sorted(apk_dir.glob("*.apk"))]
        if sha_list:
            keep = set(sha_list)
            tasks = [t for t in tasks if t[0] in keep]
        log.info("[%s] tim thay %d APK da obfuscate", tag, len(tasks))

    tasks = [(s, p) for s, p in tasks if Path(p).exists()]
    return tasks, manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 - trich feature tinh")
    ap.add_argument("--tag", default="clean",
                    help="'clean' hoac ten ky thuat (T1_trivial, ...)")
    ap.add_argument("--manifest", type=Path, default=config.MANIFEST_CSV)
    ap.add_argument("--sha-list", type=Path, default=None,
                    help="mac dinh: splits/extract_sha256.txt khi tag=clean")
    ap.add_argument("--apk-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=int, default=config.EXTRACT_TIMEOUT_S)
    ap.add_argument("--chunk", type=int, default=config.EXTRACT_CHUNK)
    ap.add_argument("--with-g5", action="store_true",
                    help="bat nhom FCG - dat gap ~5-20 lan, chi bat khi Bang B doi")
    ap.add_argument("--limit", type=int, default=0, help="chi de thu nhanh")
    args = ap.parse_args()

    config.ensure_dirs()
    manifest = pd.read_csv(args.manifest, dtype={"sha256": str})

    sha_list = None
    if args.sha_list:
        sha_list = read_sha_list(args.sha_list)
    elif args.tag == "clean":
        default = config.SPLIT_DIR / "extract_sha256.txt"
        if default.exists():
            sha_list = read_sha_list(default)
            log.info("Dung danh sach mac dinh: %s (%d sha)", default, len(sha_list))
        else:
            log.warning("Chua chay src.split - se trich TOAN BO manifest (%d APK). "
                        "PLAN muc 4 noi ro khong lam vay.", len(manifest))

    tasks, meta = build_tasks(args.tag, manifest, sha_list, args.apk_dir)
    if args.limit:
        tasks = tasks[:args.limit]

    log.info("[%s] %d APK | %d worker | timeout %ds | G5=%s",
             args.tag, len(tasks), args.workers, args.timeout, args.with_g5)
    with timed(log, f"trich feature [{args.tag}]"):
        run_extraction(tasks, args.tag, args.workers, args.with_g5,
                       args.timeout, args.chunk, meta)


if __name__ == "__main__":
    main()
