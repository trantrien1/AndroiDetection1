"""Phase 1 - lay CICMalDroid 2020 va sinh manifest.csv.

QUAN TRONG - URL trong PLAN muc 3 da chet.

    https://cicresearch.ca/CICDataset/MalDroid-2020/Dataset/APKs/

URL do gio 302 ve trang gioi thieu datasets cua UNB, va CIC da dat toan bo
dataset sau MOT FORM DANG KY (ho ten, email, to chuc, chuc danh, quoc gia).
Khong con duong tai an danh nao, nen module nay KHONG THE tu tai dataset ve.

Quy trinh thuc te bay gio:

  1. Vao https://www.unb.ca/cic/datasets/maldroid-2020.html -> "Download the
     dataset", dien form, lay duong dan tai.
  2. Tai 5 file zip theo category ve mot thu muc (vi du tren Drive de khong
     phai tai lai moi session).
  3. Chi cho module nay thu muc do:

         python -m src.download --zip-dir /content/drive/MyDrive/maldroid_zips

     Hoac neu da giai nen san thanh apks/<category>/*.apk:

         python -m src.download --manifest-only

Neu mot ngay nao do CIC mo lai duong tai truc tiep, `--url-base` van dung duoc
va moi file tai ve deu bi kiem tra magic byte truoc khi giai nen.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from . import config
from .utils import human, setup_logging, sha256_file, timed

log = setup_logging("download")

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# Quyen tai cua CIC gan voi COOKIE PHIEN dat luc submit form, khong phai voi
# tham so ?t= tren URL. Thieu cookie thi server tra 403 kem dung mot dong:
# "Registration required. Please go back and register."
COOKIE_ENV = "CIC_COOKIE"


def _headers(cookie: str | None = None, extra: dict | None = None) -> dict:
    h = {"User-Agent": USER_AGENT}
    if cookie:
        h["Cookie"] = cookie
    if extra:
        h.update(extra)
    return h


DATASET_PAGE = "https://www.unb.ca/cic/datasets/maldroid-2020.html"

MANUAL_STEPS = f"""
Cach lay dataset (CIC yeu cau dien form, khong con tai an danh duoc):

  1. Mo {DATASET_PAGE}
     bam "Download the dataset", dien form dang ky.
  2. Tai 5 file zip theo category ve mot thu muc - nen de tren Drive de khong
     phai lam lai moi session Colab.
  3. Chay lai voi thu muc do:

       python -m src.download --zip-dir /duong/dan/toi/thu/muc/zip

Neu da tu giai nen thanh apks/<category>/*.apk thi chi can:

       python -m src.download --manifest-only

Duong nhanh hon neu khong muon tai ve may roi upload lai: lay cookie phien tu
trinh duyet da dien form (DevTools > Application > Cookies > cicresearch.ca,
copy ca cap PHPSESSID=...) roi tai thang bang bang thong cua Colab:

       CIC_COOKIE='PHPSESSID=...' python -m src.download \\
           --url-base 'https://cicresearch.ca/CICDataset/MalDroid-2020/browse.php?t=...'

Cookie chi song mot phien va KHONG duoc commit vao repo.
"""

# Zip bat dau bang mot trong ba chu ky nay (local file / EOCD / spanned).
ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


# --------------------------------------------------------------------------
# Kiem tra file tai ve
# --------------------------------------------------------------------------
def looks_like_zip(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4).startswith(ZIP_MAGIC)
    except OSError:
        return False


def describe_not_zip(path: Path) -> str:
    """Noi ro file tai ve that su la gi.

    Truoc day cho nay im lang: trang HTML duoc luu thanh .zip va loi chi lo ra
    o buoc giai nen duoi dang "File is not a zip file", khong he nhac den
    nguyen nhan that la server tra ve form dang ky.
    """
    try:
        head = path.read_bytes()[:4096].decode("utf-8", "replace")
    except OSError as e:
        return f"khong doc duoc file: {e}"

    low = head.lower()
    if "dataset download form" in low or "please fill in the information" in low:
        return ("server tra ve FORM DANG KY cua CIC chu khong phai file zip. "
                "Dataset da bi dat sau form, khong tai an danh duoc nua.")
    if "<html" in low or "<!doctype html" in low:
        title = re.search(r"<title[^>]*>(.*?)</title>", head, re.I | re.S)
        t = title.group(1).strip()[:80] if title else "(khong co title)"
        return f"server tra ve trang HTML, khong phai zip. Title: {t!r}"
    return f"khong phai zip. 80 byte dau: {head[:80]!r}"


# --------------------------------------------------------------------------
# Tim zip da tai san
# --------------------------------------------------------------------------
def find_local_zip(zip_dir: Path, category: str) -> Path | None:
    """Tim file zip cua mot category trong thu muc nguoi dung chi dinh.

    Khong ep dung ten "Benign.zip": CIC dat ten khac nhau giua cac ban phat
    hanh va nguoi tai ve hay doi ten. Khop khong phan biet hoa thuong.
    """
    if not zip_dir.is_dir():
        return None
    cat = category.lower()
    exact = zip_dir / f"{category}.zip"
    if exact.is_file():
        return exact
    for p in sorted(zip_dir.glob("*.zip")):
        if cat in p.stem.lower():
            return p
    return None


# --------------------------------------------------------------------------
# Tai (chi dung duoc neu CIC mo lai duong truc tiep)
# --------------------------------------------------------------------------
def discover_zips(base_url: str, cookie: str | None = None) -> dict[str, str]:
    """Doc trang liet ke cua CIC, tra ve {category: url}.

    Hoat dong voi ca directory listing lan trang browse.php co token. Quyen
    truy cap gan voi cookie phien chu khong voi tham so ?t= tren URL, nen
    thieu cookie thi server tra 403 "Registration required".

    Tra ve dict rong neu trang khong co link zip nao - KHONG doan ten file,
    vi doan ten chinh la cai dan den chuyen luu trang HTML thanh .zip.
    """
    try:
        req = urllib.request.Request(base_url, headers=_headers(cookie))
        with urllib.request.urlopen(req, timeout=60) as r:
            final_url = r.geturl()
            html = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read(512).decode("utf-8", "replace")
        except Exception:
            pass
        if e.code == 403 or "registration required" in body.lower():
            log.error("Server tu choi (%s). CIC doi COOKIE PHIEN cua trinh duyet "
                      "da dien form - tham so ?t= tren URL khong du.", e.code)
            log.error("Lay cookie: DevTools > Application > Cookies > "
                      "cicresearch.ca, copy ca cap PHPSESSID=... roi dat vao "
                      "bien moi truong %s.", COOKIE_ENV)
        else:
            log.warning("Khong doc duoc trang liet ke: %s", e)
        return {}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.warning("Khong doc duoc trang liet ke: %s", e)
        return {}

    if "registration required" in html.lower():
        log.error("Server tra ve 'Registration required' - cookie phien thieu "
                  "hoac da het han. Dat lai %s.", COOKIE_ENV)
        return {}

    hrefs = re.findall(r'href=["\']([^"\']+\.zip)["\']', html, flags=re.I)
    if not hrefs:
        log.warning("Trang nay khong co link .zip nao.")
        return {}

    found: dict[str, str] = {}
    for href in hrefs:
        name = urllib.parse.unquote(href.rsplit("/", 1)[-1])[:-4].lower()
        for cat in config.CATEGORIES:
            if cat.lower() in name:
                # urljoin xu ly dung ca duong dan tuong doi lan query string,
                # thu ma phep noi chuoi tho truoc day lam sai.
                found.setdefault(cat, urllib.parse.urljoin(final_url, href))
                break
    log.info("Tim thay %d/%d category tren server", len(found), len(config.CATEGORIES))
    return found


def download(url: str, dest: Path, retries: int = 3,
             cookie: str | None = None) -> Path:
    """Tai co resume, va TU CHOI nhan file khong phai zip.

    Kiem tra hai lop: Content-Type tra ve tu server, va magic byte cua file
    sau khi tai xong. Lop thu hai la lop that su quan trong - server co the
    tra ve text/html kem status 200 ma khong noi gi.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = _headers(cookie)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                ctype = r.headers.get_content_type()
                if ctype.startswith("text/"):
                    body = r.read(4096).decode("utf-8", "replace")
                    part.unlink(missing_ok=True)
                    low = body.lower()
                    hint = ("FORM DANG KY cua CIC" if "dataset download form" in low
                            else "yeu cau DANG KY (thieu cookie phien)"
                            if "registration required" in low
                            else f"noi dung {ctype}")
                    raise RuntimeError(
                        f"{url}\n  server tra ve {hint}, khong phai zip.\n{MANUAL_STEPS}")

                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
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

            if not looks_like_zip(part):
                why = describe_not_zip(part)
                part.unlink(missing_ok=True)
                raise RuntimeError(f"{url}\n  {why}\n{MANUAL_STEPS}")

            part.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            log.warning("  lan %d that bai: %s", attempt, e)
            if attempt == retries:
                raise
    return dest


def extract_zip(zip_path: Path, out_dir: Path) -> int:
    """Giai nen, lam phang moi APK vao out_dir. Tra ve so APK."""
    if not looks_like_zip(zip_path):
        raise RuntimeError(f"{zip_path}\n  {describe_not_zip(zip_path)}\n{MANUAL_STEPS}")

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
    if n == 0:
        log.warning("%s khong chua file .apk nao", zip_path.name)
    return n


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def build_manifest(apk_root: Path = config.APK_DIR,
                   out_csv: Path = config.MANIFEST_CSV) -> int:
    """Sinh manifest.csv: sha256, category, label, path, size (PLAN muc 3).

    Hash lai moi file la buoc cham nhat o day, nhung sha256 la khoa duy nhat
    noi APK sach voi APK da obfuscate nen khong bo qua duoc. Neu da co manifest
    cu, tai su dung hash theo (path, size) de khong hash lai.
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
                dupes += 1          # CIC co mot so mau trung giua cac category
                continue
            seen.add(sha)
            rows.append({
                "sha256": sha, "category": cat,
                "label": config.category_to_label(cat),
                "path": str(p), "size": size,
            })

    if dupes:
        log.warning("Bo %d APK trung sha256 giua cac category", dupes)
    if not rows:
        raise RuntimeError(
            f"Khong tim thay APK nao duoi {apk_root}.\n{MANUAL_STEPS}")

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
    if n_benign < config.N_TEST // 2:
        log.warning("Chi co %d benign - it hon %d ma tap test can. Tap test se "
                    "bi lech, xem canh bao cua src.split.",
                    n_benign, config.N_TEST // 2)
    return len(rows)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 1 - lay CICMalDroid 2020 va sinh manifest.csv",
        epilog=MANUAL_STEPS, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip-dir", type=Path, default=None,
                    help="thu muc chua 5 file zip da tai thu cong (khuyen dung)")
    ap.add_argument("--url-base", default=config.CIC_BASE_URL,
                    help="trang liet ke cua CIC, vi du browse.php?t=... sau khi dien form")
    ap.add_argument("--cookie", default=os.environ.get(COOKIE_ENV),
                    help=f"cookie phien, vi du 'PHPSESSID=...'. Mac dinh lay tu "
                         f"bien moi truong {COOKIE_ENV}. KHONG commit vao repo.")
    ap.add_argument("--categories", nargs="*", default=config.CATEGORIES)
    ap.add_argument("--manifest-only", action="store_true",
                    help="APK da giai nen san thanh apks/<category>/*.apk")
    ap.add_argument("--list", action="store_true", help="thu doc index cua server")
    ap.add_argument("--keep-zip", action="store_true",
                    help="giu zip sau khi giai nen (mac dinh xoa de tiet kiem disk)")
    args = ap.parse_args()

    config.ensure_dirs()

    if args.list:
        urls = discover_zips(args.url_base, args.cookie)
        if not urls:
            log.error("Server khong cho liet ke thu muc.")
            print(MANUAL_STEPS)
            return
        for cat, url in urls.items():
            print(f"{cat:10s} {url}")
        return

    if not args.manifest_only:
        zip_dir = args.zip_dir or config.ZIP_DIR
        urls: dict[str, str] = {}
        failed: list[str] = []

        for cat in args.categories:
            out_dir = config.APK_DIR / cat
            if out_dir.is_dir() and any(out_dir.glob("*.apk")):
                log.info("%s: da giai nen, bo qua", cat)
                continue

            zip_path = find_local_zip(zip_dir, cat)

            # Don xac file hong con lai tu lan chay truoc - truoc day file nay
            # nam ly va lam moi lan chay sau deu chet o zipfile.
            if zip_path and not looks_like_zip(zip_path):
                log.warning("%s: %s khong phai zip (%s) - xoa",
                            cat, zip_path.name, describe_not_zip(zip_path))
                zip_path.unlink(missing_ok=True)
                zip_path = None

            if zip_path is None:
                if args.zip_dir:
                    log.error("%s: khong tim thay zip trong %s", cat, zip_dir)
                    failed.append(cat)
                    continue
                if not urls:
                    urls = discover_zips(args.url_base, args.cookie)
                    if not urls:
                        log.error("Khong lay duoc danh sach file tu server.")
                        print(MANUAL_STEPS)
                        raise SystemExit(2)
                url = urls.get(cat)
                if not url:
                    log.error("%s: server khong co file tuong ung", cat)
                    failed.append(cat)
                    continue
                zip_path = config.ZIP_DIR / f"{cat}.zip"
                with timed(log, f"tai {cat}"):
                    download(url, zip_path, cookie=args.cookie)

            with timed(log, f"giai nen {cat}"):
                log.info("%s: %d APK", cat, extract_zip(zip_path, config.APK_DIR / cat))
            if not args.keep_zip and not args.zip_dir:
                zip_path.unlink(missing_ok=True)

        if failed:
            log.error("Thieu category: %s", failed)
            print(MANUAL_STEPS)
            raise SystemExit(2)

    with timed(log, "sinh manifest.csv"):
        build_manifest()


if __name__ == "__main__":
    main()
