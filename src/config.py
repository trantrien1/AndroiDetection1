"""Cau hinh toan cuc.

Moi artifact co duong dan tuong minh (PLAN muc 8: "khong cache ngam").
Hai bien moi truong dieu khien layout de cung mot code chay duoc ca tren Colab
lan tren may local:

  APKROB_WORK     storage ben vung  -> Google Drive tren Colab. Song sot qua session.
  APKROB_SCRATCH  storage tam       -> /content tren Colab. Bi xoa moi session.

APK goc va APK da obfuscate nam o SCRATCH (qua nang cho Drive).
Manifest / feature / model / ket qua nam o WORK.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Seed: PLAN muc 8 - dat cung 42, log vao moi file ket qua.
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Duong dan goc
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

WORK = Path(os.environ.get("APKROB_WORK", REPO_ROOT / "data" / "work"))
SCRATCH = Path(os.environ.get("APKROB_SCRATCH", REPO_ROOT / "data" / "scratch"))

# --- ben vung (Drive) -----------------------------------------------------
MANIFEST_CSV = WORK / "manifest.csv"
SPLIT_DIR = WORK / "splits"
FEATURES_DIR = WORK / "features"
MODELS_DIR = WORK / "models"
RESULTS_DIR = WORK / "results"
LOGS_DIR = WORK / "logs"

# --- tam (bi xoa moi session) ---------------------------------------------
APK_DIR = SCRATCH / "apks"              # apks/<category>/<file>.apk
APK_OBF_DIR = SCRATCH / "apks_obf"      # apks_obf/<technique>/<sha256>.apk
OBF_WORK_DIR = SCRATCH / "obf_work"     # thu muc lam viec cua Obfuscapk
ZIP_DIR = SCRATCH / "zips"

# Tien do obfuscation phai ben vung de resume sau khi Colab timeout.
OBF_PROGRESS_JSON = WORK / "obf_progress.json"

ALL_DIRS = [
    WORK, SCRATCH, SPLIT_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR,
    APK_DIR, APK_OBF_DIR, OBF_WORK_DIR, ZIP_DIR,
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
CIC_BASE_URL = "https://cicresearch.ca/CICDataset/MalDroid-2020/Dataset/APKs/"

CATEGORIES = ["Benign", "Adware", "Banking", "SMS", "Riskware"]
BENIGN_CATEGORY = "Benign"


def category_to_label(category: str) -> int:
    """Nhi phan truoc, da lop de sau (PLAN muc 3)."""
    return 0 if category.strip().lower() == BENIGN_CATEGORY.lower() else 1


# --------------------------------------------------------------------------
# Kich thuoc split (PLAN muc 4 - ngan sach trich feature)
#
# PLAN muc 4 cap ngan sach 6.000 train + 500 test va khong nhac val, trong khi
# muc 5 yeu cau split 60/20/20 train/val/test. Vi muc 5 da cam tuning
# hyperparameter, val chi con vai tro sanity-check, nen mac dinh o day la
# 6.000 / 1.000 / 500 -> 7.500 APK sach thay vi 6.500. Doi lai bang co CLI.
# --------------------------------------------------------------------------
N_TRAIN = 6000
N_VAL = 1000
N_TEST = 500

# Subsample de obfuscate (PLAN muc 6): 250 benign + 250 malware chia deu
# 4 category malware. Bang dung N_TEST vi tap test vua du 500.
N_OBF_BENIGN = 250
N_OBF_MALWARE = 250

# --------------------------------------------------------------------------
# Nhom ky thuat obfuscation (PLAN muc 6)
#
# Luu y: Obfuscapk tu dong them Rebuild / NewAlignment / NewSignature vao cuoi
# chuoi neu thieu, nen T2..T6 deu chua san T1. Do la dieu mong muon: T1 chinh
# la control de tru ra anh huong cua viec dong goi lai.
# --------------------------------------------------------------------------
TECHNIQUES: dict[str, list[str]] = {
    "T1_trivial": ["Rebuild", "NewAlignment", "NewSignature"],
    "T2_rename": ["ClassRename", "MethodRename", "FieldRename"],
    "T3_string": ["ConstStringEncryption", "ResStringEncryption"],
    "T4_asset": ["AssetEncryption", "LibEncryption"],
    "T5_cfg": ["Goto", "ArithmeticBranch", "Nop", "Reorder"],
    "T6_reflection": ["CallIndirection", "Reflection", "AdvancedReflection"],
}

TECHNIQUE_ORDER = list(TECHNIQUES.keys())

# Nhung obfuscator luon duoc Obfuscapk gan vao cuoi chuoi.
TRIVIAL_TAIL = ["Rebuild", "NewAlignment", "NewSignature"]

# --------------------------------------------------------------------------
# Nhom feature (PLAN muc 4)
# --------------------------------------------------------------------------
GROUPS = ["G1", "G2", "G3", "G4", "G5"]
GROUPS_DEFAULT = ["G1", "G2", "G3", "G4"]  # G5 hoan sang pass sau

GROUP_LABELS = {
    "G1": "G1 Manifest",
    "G2": "G2 API",
    "G3": "G3 Opcode",
    "G4": "G4 String",
    "G5": "G5 FCG",
}

# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------
EXTRACT_TIMEOUT_S = 60      # Androguard treo tren APK hong
EXTRACT_CHUNK = 500         # checkpoint moi 500 APK
OBF_TIMEOUT_S = 900         # Obfuscapk cham; 15 phut/APK la tran cung
OBF_WORKERS = 4


def technique_feature_path(technique: str) -> Path:
    return FEATURES_DIR / f"features_obf_{technique}.parquet"


CLEAN_FEATURE_PATH = FEATURES_DIR / "features_clean.parquet"
