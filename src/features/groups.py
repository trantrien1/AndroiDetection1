"""Dinh nghia 5 nhom feature G1-G5 (PLAN muc 4).

Moi feature key co dang "<GROUP>:<ten>". Tien to nay la thu duy nhat noi
vector hoa voi Bang B o Phase 5: vectorizer cat cot theo tien to de train
model chi-mot-nhom, nen khong duoc doi dinh dang key.

| Nhom | Noi dung                    | Nguon Androguard | Du doan do nhay |
|------|-----------------------------|------------------|-----------------|
| G1   | manifest, permission, SDK   | APK              | Thap            |
| G2   | Android framework API calls | DEX              | Trung binh      |
| G3   | phan bo opcode + 2-gram     | DEX              | Thap            |
| G4   | string: URL/IP/phone/b64    | DEX              | Rat cao         |
| G5   | call graph (hoan pass sau)  | Analysis         | ?               |
"""
from __future__ import annotations

import re

SEP = ":"

G1, G2, G3, G4, G5 = "G1", "G2", "G3", "G4", "G5"


def key(group: str, name: str) -> str:
    return f"{group}{SEP}{name}"


def group_of(feature_key: str) -> str:
    return feature_key.split(SEP, 1)[0]


# --------------------------------------------------------------------------
# G2 - chi giu API thuoc framework, bo code cua chinh app
#
# Lop do app tu dinh nghia bi ClassRename doi ten -> se sinh ra hang nghin
# token rac chi xuat hien o dung 1 APK. Loc o day cho re hon loc o min_df.
# --------------------------------------------------------------------------
FRAMEWORK_PREFIXES = (
    "Landroid/", "Landroidx/", "Ljava/", "Ljavax/", "Ldalvik/",
    "Lorg/apache/", "Lorg/json/", "Lorg/w3c/", "Lorg/xml/", "Lorg/xmlpull/",
    "Lcom/android/", "Lcom/google/android/", "Ljunit/", "Lkotlin/",
)

# Duong dan gia tri cao - luon giu rieng ke ca khi min_df cat.
SENSITIVE_API_HINTS = (
    "sendTextMessage", "getDeviceId", "getSubscriberId", "getSimSerialNumber",
    "getLine1Number", "getInstalledPackages", "getRunningTasks",
    "exec", "loadLibrary", "loadClass", "getDeclaredMethod", "invoke",
    "openConnection", "getOutputStream", "setComponentEnabledSetting",
    "getLastKnownLocation", "requestLocationUpdates", "query", "getContentResolver",
    "createFromPdu", "abortBroadcast", "setWifiEnabled", "killBackgroundProcesses",
)

# invoke-virtual v1, Landroid/telephony/SmsManager;->sendTextMessage(...)V
INVOKE_TARGET_RE = re.compile(r"(L[A-Za-z0-9_$/\-]+;)->([A-Za-z0-9_$<>]+)")


def is_framework_class(clazz: str) -> bool:
    return clazz.startswith(FRAMEWORK_PREFIXES)


# --------------------------------------------------------------------------
# G4 - string
# --------------------------------------------------------------------------
URL_RE = re.compile(r"https?://[^\s\"'<>\\]{4,}", re.I)
IP_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
                   r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b")
PHONE_RE = re.compile(r"(?<!\d)\+?\d[\d\-\s().]{7,16}\d(?!\d)")
BASE64_RE = re.compile(r"^[A-Za-z0-9+/]{24,}={0,2}$")
HEXBLOB_RE = re.compile(r"^(?:[0-9a-fA-F]{2}){16,}$")
DOMAIN_RE = re.compile(r"https?://([^/\s:\"'<>\\]+)", re.I)

# Tu khoa co y nghia hanh vi. Dem so string chua tu khoa -> mot vector nho,
# on dinh hon la bo tui toan bo string (von no ra hang tram nghin token).
STRING_KEYWORDS = (
    "sms", "smsto", "sendto", "pdus", "shell", "/system/bin", "/system/xbin",
    "su", "busybox", "root", "superuser", "chmod", "mount", "dexopt",
    ".dex", ".jar", ".so", "base64", "aes", "des", "rc4", "md5", "sha1",
    "http://", "https://", "socket", "telnet", "ftp", "bot", "cmd",
    "install", "uninstall", "package:", "content://", "market://",
    "android.intent.action.boot_completed", "device_admin",
    "getexternalstorage", "sdcard", "crypt", "decrypt", "payload",
)

# Chuoi dai bao nhieu thi moi dang tinh entropy. Chuoi ngan cho entropy nhieu.
MIN_ENTROPY_LEN = 12
HIGH_ENTROPY_THRESHOLD = 4.5   # bit/ky tu - nguong quen dung cho chuoi da ma hoa


# --------------------------------------------------------------------------
# G1 - manifest
#
# Khong giu whitelist permission cung: min_df o vectorizer se cat permission
# hiem. Nhung so activity/service/receiver va co exported thi luon giu.
# --------------------------------------------------------------------------
COMPONENT_TAGS = ("activity", "activity-alias", "service", "receiver", "provider")
ANDROID_NS = "{http://schemas.android.com/apk/res/android}"


# --------------------------------------------------------------------------
# G3 - opcode
#
# Opcode Dalvik co ~230 ma. 2-gram trong pham vi mot method -> khong bao gio
# bat cau qua ranh gioi method (PLAN muc 4: "muc method roi tong hop").
# Cap so 2-gram giu lai de tranh no bo nho tren APK khong lo.
# --------------------------------------------------------------------------
MAX_OPCODE_2GRAMS = 4000
MAX_API_TOKENS = 4000
MAX_DOMAIN_TOKENS = 400


def summary_table() -> str:
    rows = [
        ("G1", "permissions, intents, component counts, SDK", "APK", "Thap"),
        ("G2", "Android framework API calls (count)", "DEX", "Trung binh"),
        ("G3", "opcode histogram + 2-gram", "DEX", "Thap"),
        ("G4", "URL/IP/phone/base64/entropy/keyword", "DEX", "Rat cao"),
        ("G5", "call graph structure (opt-in)", "Analysis", "?"),
    ]
    out = ["| Nhom | Noi dung | Nguon | Du doan do nhay |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    return "\n".join(out)
