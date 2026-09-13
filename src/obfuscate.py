"""Phase 4 - pipeline obfuscation bang Obfuscapk (PLAN muc 6).

Ba quy tac cua PLAN duoc cai cung vao day:
  1. CHI obfuscate tap test. Module nay chi doc splits/obf_sha256.txt, von
     duoc cat ra tu test_sha256.txt. Khong co duong nao de no cham vao train.
  2. Obfuscate CA benign lan malware. Tap con da la 250/250 tu src.split.
  3. Verify sau khi obfuscate. APK nao apksigner reject hoac Androguard khong
     parse duoc thi bi loai, va ti le hong duoc ghi lai theo tung ky thuat -
     ban than no la mot ket qua dang bao cao (Bang A).

Chi phi: 500 APK x 6 nhom x ~45s / 4 worker ~ 9-10 gio -> chia 2 session.

Chay:
    python -m src.obfuscate --smoke-test                  # Phase 0, chay TRUOC
    python -m src.obfuscate --techniques T1_trivial T2_rename T3_string
    python -m src.obfuscate --techniques T4_asset T5_cfg T6_reflection
    python -m src.obfuscate --report
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from . import config
from .utils import (atomic_write_json, human, read_json, read_sha_list,
                    seed_stamp, setup_logging, sha256_file, timed)

log = setup_logging("obfuscate")

_progress_lock = threading.Lock()


# --------------------------------------------------------------------------
# Toolchain
# --------------------------------------------------------------------------
REQUIRED_TOOLS = {
    "apktool": "APKTOOL_PATH",
    "apksigner": "APKSIGNER_PATH",
    "zipalign": "ZIPALIGN_PATH",
}


def check_toolchain() -> dict[str, str | None]:
    """Obfuscapk goi apktool/apksigner/zipalign qua PATH hoac bien moi truong.

    PLAN muc 2 cai zipalign va apksigner nhung KHONG cai apktool - thieu no thi
    moi ky thuat deu fail ngay buoc decompile. Kiem tra o day de biet ngay.
    """
    found: dict[str, str | None] = {}
    for tool, env in REQUIRED_TOOLS.items():
        path = os.environ.get(env) or shutil.which(tool)
        found[tool] = path
        if path:
            log.info("  %-10s %s", tool, path)
        else:
            log.error("  %-10s KHONG TIM THAY (dat %s hoac them vao PATH)", tool, env)

    ok, reason = check_obfuscapk()
    found["obfuscapk"] = "ok" if ok else None
    if ok:
        log.info("  %-10s ok", "obfuscapk")
    else:
        log.error("  %-10s HONG\n%s", "obfuscapk", reason)
    return found


def obfuscapk_cmd() -> list[str]:
    exe = shutil.which("obfuscapk")
    return [exe] if exe else [sys.executable, "-m", "obfuscapk.cli"]


def obfuscapk_env() -> dict[str, str]:
    """Moi truong cho subprocess Obfuscapk.

    Obfuscapk KHONG co tren PyPI va cung khong co setup.py o goc repo, nen
    khong pip install duoc bang bat ky cach nao - phai clone roi dua thu muc
    src/ cua no vao PYTHONPATH. Dat OBFUSCAPK_SRC tro toi thu muc do.
    """
    env = os.environ.copy()
    src = env.get("OBFUSCAPK_SRC")
    if src:
        parts = [src] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def check_obfuscapk() -> tuple[bool, str]:
    """Chay `obfuscapk --help`. Hai kieu hong da biet duoc dich sang cach sua.

    Khong co buoc nay thi ca hai loi duoi day chi lo ra sau khi da tai xong
    dataset va trich xong feature - tuc la muon vai gio.
    """
    try:
        r = subprocess.run(obfuscapk_cmd() + ["--help"], capture_output=True,
                           timeout=180, env=obfuscapk_env())
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"khong chay duoc: {e}"
    if r.returncode == 0:
        return True, "ok"

    err = (r.stderr or r.stdout or b"").decode("utf-8", "replace")
    if "No module named 'obfuscapk'" in err or "No module named obfuscapk" in err:
        return False, (
            "khong import duoc obfuscapk. No khong co tren PyPI ('pip install "
            "obfuscapk' luon that bai) va cung khong co setup.py o goc repo. "
            "Clone roi tro OBFUSCAPK_SRC vao thu muc src/ cua no:\n"
            "  git clone --depth 1 https://github.com/ClaudiuGeorgiu/Obfuscapk.git /content/Obfuscapk\n"
            "  export OBFUSCAPK_SRC=/content/Obfuscapk/src")
    if "No module named 'imp'" in err:
        return False, (
            "Yapsy 1.12.2 (ban moi nhat tren PyPI, 2019) dung module 'imp', von "
            "da bi XOA khoi Python 3.12. Ban vo master cua Yapsy da chuyen sang "
            "importlib nhung chua bao gio duoc phat hanh. Cai tu git:\n"
            "  pip install 'yapsy @ git+https://github.com/tibonihoo/yapsy.git"
            "@master#subdirectory=package'")
    return False, f"rc={r.returncode}: {err.strip()[:300]}"


# --------------------------------------------------------------------------
# Verify
# --------------------------------------------------------------------------
def verify_apk(path: Path, apksigner: str | None = None) -> tuple[bool, str]:
    """Quy tac 3: APK sau obfuscate phai ky hop le VA Androguard parse duoc.

    Neu chi kiem tra apksigner thi van lot nhung APK ky dung nhung dex hong -
    va dung nhung APK do moi lam Phase 5 ra so lech ma khong ai biet vi sao.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False, "khong co file output"

    apksigner = apksigner or os.environ.get("APKSIGNER_PATH") or shutil.which("apksigner")
    if apksigner:
        try:
            r = subprocess.run([apksigner, "verify", str(path)],
                               capture_output=True, timeout=120)
            if r.returncode != 0:
                msg = (r.stderr or r.stdout).decode("utf-8", "replace").strip()
                return False, f"apksigner: {msg[:150]}"
        except subprocess.TimeoutExpired:
            return False, "apksigner timeout"
        except OSError as e:
            log.warning("khong chay duoc apksigner (%s) - bo qua buoc nay", e)

    try:
        from androguard.core.apk import APK
    except ImportError:
        try:
            from androguard.core.bytecodes.apk import APK  # type: ignore
        except ImportError:
            return True, "ok (khong co androguard de kiem tra)"
    try:
        a = APK(str(path))
        if not a.is_valid_APK():
            return False, "androguard: APK khong hop le"
        if not list(a.get_all_dex()):
            return False, "androguard: khong co dex"
    except Exception as e:
        return False, f"androguard: {type(e).__name__}: {e}"[:150]
    return True, "ok"


# --------------------------------------------------------------------------
# Chay mot APK
# --------------------------------------------------------------------------
def obfuscate_one(src_apk: Path, out_apk: Path, obfuscators: list[str],
                  work_dir: Path, timeout: int = config.OBF_TIMEOUT_S,
                  ignore_libs: bool = False) -> dict:
    t0 = time.time()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_apk.parent.mkdir(parents=True, exist_ok=True)

    cmd = obfuscapk_cmd()
    for o in obfuscators:
        cmd += ["-o", o]
    cmd += ["-w", str(work_dir), "-d", str(out_apk)]
    if ignore_libs:
        cmd.append("--ignore-libs")
    cmd.append(str(src_apk))

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout,
                           env=obfuscapk_env())
        rc, err = r.returncode, (r.stderr or b"").decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        rc, err = -1, f"timeout>{timeout}s"
    except OSError as e:
        rc, err = -1, f"khong chay duoc obfuscapk: {e}"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    seconds = round(time.time() - t0, 1)
    if rc != 0:
        out_apk.unlink(missing_ok=True)
        return {"status": "fail", "reason": f"obfuscapk rc={rc}: {err.strip()[:200]}",
                "seconds": seconds}

    ok, reason = verify_apk(out_apk)
    if not ok:
        out_apk.unlink(missing_ok=True)
        return {"status": "fail", "reason": reason, "seconds": seconds}

    out_sha = sha256_file(out_apk)
    rec = {"status": "ok", "reason": "", "seconds": seconds,
           "out_sha256": out_sha, "out_size": out_apk.stat().st_size}
    if out_sha == sha256_file(src_apk):
        # PLAN muc 8: "kiem tra APK co that su bi obfuscate khong".
        rec["status"] = "fail"
        rec["reason"] = "output trung byte voi input - khong co gi bi doi"
        out_apk.unlink(missing_ok=True)
    return rec


# --------------------------------------------------------------------------
# Tien do - phai ben vung de resume sau timeout Colab
# --------------------------------------------------------------------------
def load_progress() -> dict:
    return read_json(config.OBF_PROGRESS_JSON, default={}) or {}


def save_progress(progress: dict) -> None:
    with _progress_lock:
        atomic_write_json(config.OBF_PROGRESS_JSON, progress)


# --------------------------------------------------------------------------
def technique_chain(technique: str) -> list[str]:
    """Chuoi obfuscator day du cho mot nhom ky thuat.

    Obfuscapk tu gan Rebuild/NewAlignment/NewSignature vao cuoi chuoi neu thieu.
    O day ta ghi ra tuong minh thay vi dua vao hanh vi ngam do - va dong thoi
    noi ro vi sao T1 la control dung: moi nhom T2-T6 deu chua tron ven T1,
    nen hieu so giua chung la phan do RIENG ky thuat do gay ra.
    """
    chain = list(config.TECHNIQUES[technique])
    for t in config.TRIVIAL_TAIL:
        if t not in chain:
            chain.append(t)
    return chain


def run_technique(technique: str, tasks: list[tuple[str, Path]], progress: dict,
                  workers: int, timeout: int, ignore_libs: bool,
                  retry_failed: bool = False, flush_every: int = 10) -> dict:
    obfuscators = technique_chain(technique)
    out_dir = config.APK_OBF_DIR / technique
    out_dir.mkdir(parents=True, exist_ok=True)
    done = progress.setdefault(technique, {})

    pending = []
    for sha, src in tasks:
        rec = done.get(sha)
        out_apk = out_dir / f"{sha}.apk"
        if rec and rec.get("status") == "ok" and out_apk.exists():
            continue
        if rec and rec.get("status") == "fail" and not retry_failed:
            continue        # da thu va hong - ghi vao ti le hong, khong thu lai
        pending.append((sha, src, out_apk))

    log.info("[%s] %s | con %d/%d APK", technique, "+".join(obfuscators),
             len(pending), len(tasks))
    if not pending:
        return done

    counter = {"n": 0, "ok": 0, "fail": 0}
    t0 = time.time()

    def _task(item):
        sha, src, out_apk = item
        work = config.OBF_WORK_DIR / technique / sha
        rec = obfuscate_one(src, out_apk, obfuscators, work, timeout, ignore_libs)
        with _progress_lock:
            done[sha] = rec
            counter["n"] += 1
            counter["ok" if rec["status"] == "ok" else "fail"] += 1
            n = counter["n"]
        if rec["status"] == "fail":
            log.debug("[%s] %s FAIL: %s", technique, sha[:12], rec["reason"])
        if n % flush_every == 0:
            save_progress(progress)
            el = time.time() - t0
            rate = n / el if el else 0
            log.info("[%s] %d/%d ok=%d fail=%d | %.2f APK/s | ETA %.0f phut",
                     technique, n, len(pending), counter["ok"], counter["fail"],
                     rate, (len(pending) - n) / rate / 60 if rate else 0)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_task, pending))

    save_progress(progress)
    n_ok = sum(1 for r in done.values() if r.get("status") == "ok")
    log.info("[%s] xong: %d ok / %d thu (%.1f%% hong)",
             technique, n_ok, len(done), 100 * (1 - n_ok / max(1, len(done))))
    return done


# --------------------------------------------------------------------------
def broken_rate_report(progress: dict | None = None) -> pd.DataFrame:
    """Ti le APK hong theo tung ky thuat - cot cuoi cua Bang A."""
    progress = progress if progress is not None else load_progress()
    rows = []
    for tech in config.TECHNIQUE_ORDER:
        recs = progress.get(tech, {})
        if not recs:
            continue
        n = len(recs)
        n_ok = sum(1 for r in recs.values() if r.get("status") == "ok")
        secs = [r.get("seconds", 0) for r in recs.values()]
        reasons: dict[str, int] = {}
        for r in recs.values():
            if r.get("status") != "ok":
                kind = str(r.get("reason", "?")).split(":")[0][:40]
                reasons[kind] = reasons.get(kind, 0) + 1
        rows.append({
            "technique": tech,
            "obfuscators": "+".join(technique_chain(tech)),
            "n_attempted": n,
            "n_ok": n_ok,
            "n_broken": n - n_ok,
            "broken_pct": round(100 * (n - n_ok) / n, 2) if n else 0.0,
            "median_s": round(float(pd.Series(secs).median()), 1) if secs else 0.0,
            "top_reasons": "; ".join(f"{k}={v}" for k, v in
                                     sorted(reasons.items(), key=lambda x: -x[1])[:3]),
        })
    return pd.DataFrame(rows)


def smoke_test(manifest: Path) -> bool:
    """PLAN muc 2: chay Rebuild tren 1 APK mau, xac nhan output ton tai va
    apksigner verify pass. Neu buoc nay hong thi moi thu sau deu vo nghia -
    dung di tiep.
    """
    log.info("--- Kiem tra toolchain ---")
    tools = check_toolchain()
    if not tools.get("apktool"):
        log.error("Thieu apktool. Tren Colab: apt-get install -y apktool")
        return False
    if not tools.get("obfuscapk"):
        log.error("Obfuscapk chua chay duoc - xem huong dan ngay tren.")
        return False

    df = pd.read_csv(manifest, dtype={"sha256": str})
    row = df.iloc[0]
    src = Path(row["path"])
    if not src.exists():
        log.error("APK mau khong ton tai: %s", src)
        return False

    out = config.SCRATCH / "smoke" / "rebuild.apk"
    work = config.SCRATCH / "smoke" / "work"
    log.info("--- Chay Rebuild tren %s (%s) ---", src.name, human(src.stat().st_size))
    with timed(log, "smoke test"):
        rec = obfuscate_one(src, out, config.TRIVIAL_TAIL, work, timeout=600)

    if rec["status"] != "ok":
        log.error("SMOKE TEST HONG: %s", rec["reason"])
        log.error("Dung di tiep cho den khi buoc nay pass.")
        return False
    log.info("SMOKE TEST PASS: %s (%s, %.1fs)", out, human(rec["out_size"]), rec["seconds"])
    return True


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 - obfuscate tap test")
    ap.add_argument("--techniques", nargs="*", default=config.TECHNIQUE_ORDER,
                    choices=config.TECHNIQUE_ORDER)
    ap.add_argument("--manifest", type=Path, default=config.MANIFEST_CSV)
    ap.add_argument("--sha-list", type=Path,
                    default=config.SPLIT_DIR / "obf_sha256.txt")
    ap.add_argument("--workers", type=int, default=config.OBF_WORKERS)
    ap.add_argument("--timeout", type=int, default=config.OBF_TIMEOUT_S)
    ap.add_argument("--ignore-libs", action="store_true",
                    help="bo qua thu vien ben thu ba - nhanh hon, nhung T4 mat y nghia")
    ap.add_argument("--retry-failed", action="store_true",
                    help="thu lai nhung APK da fail (dung khi fail do het RAM/dut "
                         "session chu khong phai do ban than APK)")
    ap.add_argument("--max-apks", type=int, default=0, help="chi de thu nhanh")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()

    if args.smoke_test:
        sys.exit(0 if smoke_test(args.manifest) else 1)

    if args.report:
        rep = broken_rate_report()
        if rep.empty:
            log.info("Chua co tien do nao.")
        else:
            print(rep.to_string(index=False))
            rep.to_csv(config.RESULTS_DIR / "obf_broken_rate.csv", index=False)
        return

    manifest = pd.read_csv(args.manifest, dtype={"sha256": str})
    path_of = dict(zip(manifest["sha256"], manifest["path"]))

    shas = read_sha_list(args.sha_list)
    if args.max_apks:
        shas = shas[:args.max_apks]

    # Quy tac 1: kiem tra lai rang tap nay nam tron trong tap test da chot.
    test_path = config.SPLIT_DIR / "test_sha256.txt"
    if test_path.exists():
        test_set = set(read_sha_list(test_path))
        leaked = [s for s in shas if s not in test_set]
        if leaked:
            log.error("DUNG LAI: %d sha khong thuoc tap test da chot. "
                      "Obfuscate ngoai tap test la vi pham quy tac 1 cua PLAN.",
                      len(leaked))
            sys.exit(1)

    tasks = []
    for sha in shas:
        p = path_of.get(sha)
        if p and Path(p).exists():
            tasks.append((sha, Path(p)))
    if len(tasks) < len(shas):
        log.warning("%d/%d APK khong tim thay tren disk - can tai lai dataset",
                    len(shas) - len(tasks), len(shas))

    est = len(tasks) * len(args.techniques) * 45 / max(1, args.workers) / 3600
    log.info("%d APK x %d ky thuat / %d worker ~ %.1f gio",
             len(tasks), len(args.techniques), args.workers, est)

    progress = load_progress()
    for tech in args.techniques:
        with timed(log, f"obfuscate {tech}"):
            run_technique(tech, tasks, progress, args.workers, args.timeout,
                          args.ignore_libs, args.retry_failed)

    rep = broken_rate_report(progress)
    if not rep.empty:
        rep.to_csv(config.RESULTS_DIR / "obf_broken_rate.csv", index=False)
        atomic_write_json(config.RESULTS_DIR / "obf_summary.json",
                          {**seed_stamp(), "per_technique": rep.to_dict("records")})
        print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
