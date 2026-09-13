"""Tien ich dung chung: logging, hash, ghi file nguyen tu, checkpoint.

Nguyen tac (PLAN muc 8): moi buoc dai phai resume duoc; gia dinh session Colab
chet bat cu luc nao. Nghia la moi lan ghi state deu phai nguyen tu (ghi file
tmp roi os.replace) de khong bao gio doc phai file viet do dang.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import config


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                            datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(config.LOGS_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


@contextmanager
def timed(logger: logging.Logger, what: str) -> Iterator[None]:
    t0 = time.time()
    logger.info("BAT DAU %s", what)
    try:
        yield
    finally:
        logger.info("XONG   %s (%.1fs)", what, time.time() - t0)


# --------------------------------------------------------------------------
# Seed
# --------------------------------------------------------------------------
def set_seed(seed: int = config.RANDOM_SEED) -> int:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    return seed


def seed_stamp() -> dict[str, Any]:
    """Dong dau vao moi file ket qua (PLAN muc 8)."""
    return {
        "random_seed": config.RANDOM_SEED,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# --------------------------------------------------------------------------
# Hash
# --------------------------------------------------------------------------
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Ghi file nguyen tu
# --------------------------------------------------------------------------
def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=False,
                                       default=_json_default))


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"khong serialize duoc: {type(o)}")


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # File hong vi session chet giua chung -> coi nhu chua co.
        return default


# --------------------------------------------------------------------------
# Danh sach sha256 (file .txt mot dong mot hash)
# --------------------------------------------------------------------------
def write_sha_list(path: str | Path, shas: Iterable[str]) -> None:
    atomic_write_text(path, "\n".join(shas) + "\n")


def read_sha_list(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"khong tim thay danh sach sha256: {path}")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


# --------------------------------------------------------------------------
# Chunk
# --------------------------------------------------------------------------
def chunked(seq: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# --------------------------------------------------------------------------
# Shannon entropy - dung cho G4
# --------------------------------------------------------------------------
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    import math
    n = len(s)
    counts = Counter(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"
