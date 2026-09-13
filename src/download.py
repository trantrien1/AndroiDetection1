"""Phase 1 - tai CICMalDroid 2020 va sinh manifest.csv.

PLAN muc 3: chap nhan tai lai moi session, nhung manifest.csv phai nam tren
Drive vi day la thu duy nhat can giu giua cac session o phase nay.

Chay:
    python -m src.download                 # tai + giai nen + sinh manifest
    python -m src.download --manifest-only # APK da co san, chi dung lai manifest
    python -m src.download --list          # chi liet ke file zip tren server
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from . import config
from .utils import human, setup_logging, sha256_file, timed

log = setup_logging("download")

USER_AGENT = "Mozilla/5.0 (compatible; apk-robustness-research/1.0)"


# --------------------------------------------------------------------------
# Kham pha file zip tren server
# --------------------------------------------------------------------------
def discover_zips(base_url: str = config.CIC_BASE_URL) -> dict[str, str]:
    """Doc index HTML cua thu muc, tra ve {category: url}.

    Khong hardcode ten file vi CIC doi ten giua cac ban phat hanh. Neu index
    khong doc duoc thi rot ve ten doan theo category.
    """
    try:
        req = urllib.request.Request(base_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            html = r.read().decode("utf-8", errors="replace")
        hrefs = re.findall(r'href="([^"]+\.zip)"', html, flags=re.I)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.warning("Khong doc duoc index (%s) - dung ten file doan theo category", e)
        hrefs = [f"{c}.zip" for c in config.CATEGORIES]

    found: dict[str, str] = {}
    for href in hrefs:
        name = href.rsplit("/", 1)[-1]
        stem = name[:-4].lower()
        for cat in config.CATEGORIES:
            if cat.lower() in stem or stem in cat.lower():
                url = (href if href.startswith("http")
                       else base_url.rstrip("/") + "/" + href.lstrip("/"))
                found.setdefault(cat, url)
                break

    missing = [c for c in config.CATEGORIES if c not in found]
    if missing:
        log.warning("Khong khop duoc category: %s - dung ten doan", missing)
        for c in missing:
            found[c] = base_url.rstrip("/") + f"/{c}.zip"
    return found


# --------------------------------------------------------------------------
# Tai co resume
# --------------------------------------------------------------------------
def download(url: str, dest: Path, retries: int = 3) -> Path:
    """Tai co ho tro HTTP Range de resume sau khi Colab dut mang."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                # Server bo qua Range -> phai ghi lai tu dau.
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                log.info("  tai %s (da co %s, tong %s)", url.rsplit("/", 1)[-1],
                         human(have), human(total) if total else "?")
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            part.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            log.warning("  lan %d that bai: %s", attempt, e)
            if attempt == retries:
                raise
    return dest


def extract_zip(zip_path: Path, out_dir: Path) -> int:
    """Giai nen, lam phang moi APK vao out_dir. Tra ve so APK."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".apk"):
                continue
            target = out_dir / Path(info.filename).name
            if target.exists() and target.stat().st_size == info.file_size:
                n += 1
                continue
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            n += 1
    return n


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def build_manifest(apk_root: Path = config.APK_DIR,
                   out_csv: Path = config.MANIFEST_CSV) -> int:
    """Sinh manifest.csv: sha256, category, label, path, size (PLAN muc 3).

    Hash lai moi file la buoc cham nhat o day (~17k file), nhung sha256 la khoa
    duy nhat noi APK sach voi APK da obfuscate nen khong bo qua duoc. Neu da co
    manifest cu, tai su dung hash theo (path, size) de khong hash lai.
    """
    cache: dict[tuple[str, int], str] = {}
    if out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    cache[(row["path"], int(row["size"]))] = row["sha256"]
                except (KeyError, ValueError):
                    continue
        log.info("Tai lai %d hash tu manifest cu", len(cache))

    rows: list[dict] = []
    seen: set[str] = set()
    dupes = 0

    for cat in config.CATEGORIES:
        cat_dir = apk_root / cat
        if not cat_dir.is_dir():
            log.warning("Thieu thu muc category: %s", cat_dir)
            continue
        files = sorted(cat_dir.glob("*.apk"))
        log.info("%-9s %d APK", cat, len(files))
        for p in files:
            size = p.stat().st_size
            sha = cache.get((str(p), size)) or sha256_file(p)
            if sha in seen:
                # CIC co mot so mau trung giua cac category. Giu lan dau.
                dupes += 1
                continue
            seen.add(sha)
            rows.append({
                "sha256": sha,
                "category": cat,
                "label": config.category_to_label(cat),
                "path": str(p),
                "size": size,
            })

    if dupes:
        log.warning("Bo %d APK trung sha256 giua cac category", dupes)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sha256", "category", "label", "path", "size"])
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out_csv)

    n_benign = sum(1 for r in rows if r["label"] == 0)
    log.info("manifest.csv: %d APK (benign=%d malware=%d) -> %s",
             len(rows), n_benign, len(rows) - n_benign, out_csv)
    return len(rows)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 - tai CICMalDroid 2020")
    ap.add_argument("--base-url", default=config.CIC_BASE_URL)
    ap.add_argument("--categories", nargs="*", default=config.CATEGORIES)
    ap.add_argument("--manifest-only", action="store_true",
                    help="APK da giai nen san, chi dung lai manifest.csv")
    ap.add_argument("--list", action="store_true", help="chi liet ke url zip")
    ap.add_argument("--keep-zip", action="store_true",
                    help="giu zip sau khi giai nen (mac dinh xoa de tiet kiem disk)")
    args = ap.parse_args()

    config.ensure_dirs()

    if args.list:
        for cat, url in discover_zips(args.base_url).items():
            print(f"{cat:10s} {url}")
        return

    if not args.manifest_only:
        urls = discover_zips(args.base_url)
        for cat in args.categories:
            url = urls.get(cat)
            if not url:
                log.error("Khong co url cho %s", cat)
                continue
            zip_path = config.ZIP_DIR / f"{cat}.zip"
            out_dir = config.APK_DIR / cat
            if out_dir.is_dir() and any(out_dir.glob("*.apk")):
                log.info("%s: da giai nen, bo qua", cat)
                continue
            with timed(log, f"tai+giai nen {cat}"):
                if not zip_path.exists():
                    download(url, zip_path)
                log.info("%s: %d APK", cat, extract_zip(zip_path, out_dir))
                if not args.keep_zip:
                    zip_path.unlink(missing_ok=True)

    with timed(log, "sinh manifest.csv"):
        build_manifest()


if __name__ == "__main__":
    main()
