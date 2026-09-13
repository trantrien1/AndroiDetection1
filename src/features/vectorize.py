"""Dict feature -> sparse matrix. Fit CHI tren train (PLAN muc 5).

Vocabulary duoc hoc mot lan tren tap train sach roi dong bang. Val, test va
moi APK da obfuscate deu chi duoc transform. Fit lai tren tap khac la leakage,
va nang hon: neu fit lai tren APK da obfuscate thi ta dang do mot chuyen khac
han - model se thay duoc token moi do obfuscator sinh ra.

Cot duoc xep lien tiep theo nhom (G1 roi G2 roi G3...), nho vay cat mot nhom
ra chi la mot lat cat - day la thu Bang B o Phase 5 can.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .. import config
from ..utils import setup_logging
from . import groups as G

log = setup_logging("vectorize")


class GroupVectorizer:
    """DictVectorizer co y thuc ve nhom feature.

    Tham so:
      groups          nhom duoc giu lai
      min_df          bo feature xuat hien o < min_df APK trong tap train.
                      Khong co buoc nay thi 2-gram opcode va ten API sinh ra
                      hang tram nghin cot chi xuat hien dung mot lan.
      max_per_group   tran so cot moi nhom (theo document frequency giam dan)
      normalize_rows  nhom duoc chia cho tong so instruction cua APK do ->
                      G3 thanh phan bo thay vi count tho, nen so sanh duoc
                      giua APK 2k method va APK 200k method
      log1p           nen count cho cac nhom con lai
    """

    def __init__(self, groups: Sequence[str] = tuple(config.GROUPS_DEFAULT),
                 min_df: int = 5, max_per_group: int | None = 20000,
                 normalize_rows: Sequence[str] = ("G3",), log1p: bool = True):
        self.groups = list(groups)
        self.min_df = min_df
        self.max_per_group = max_per_group
        self.normalize_rows = set(normalize_rows)
        self.log1p = log1p

        self.vocabulary_: dict[str, int] = {}
        self.feature_names_: list[str] = []
        self.group_slices_: dict[str, tuple[int, int]] = {}
        self.n_features_: int = 0
        self.fitted_on_: int = 0

    # ------------------------------------------------------------------
    def fit(self, dicts: Iterable[dict[str, float]]) -> "GroupVectorizer":
        df_counts: Counter[str] = Counter()
        n_docs = 0
        for d in dicts:
            n_docs += 1
            for k in d:
                if k.split(G.SEP, 1)[0] in self.groups:
                    df_counts[k] += 1

        self.fitted_on_ = n_docs
        kept_by_group: dict[str, list[str]] = {g: [] for g in self.groups}
        for k, c in df_counts.items():
            if c >= self.min_df:
                kept_by_group[G.group_of(k)].append(k)

        names: list[str] = []
        for g in self.groups:
            ks = kept_by_group[g]
            if self.max_per_group and len(ks) > self.max_per_group:
                ks = sorted(ks, key=lambda k: (-df_counts[k], k))[:self.max_per_group]
                log.info("  %s: cat con %d cot (tran max_per_group)", g, len(ks))
            ks.sort()
            start = len(names)
            names.extend(ks)
            self.group_slices_[g] = (start, len(names))
            log.info("  %-3s %6d cot (tu %d key tho)", g, len(ks), len(kept_by_group[g]))

        self.feature_names_ = names
        self.vocabulary_ = {k: i for i, k in enumerate(names)}
        self.n_features_ = len(names)
        log.info("Vocabulary: %d cot, fit tren %d APK (min_df=%d)",
                 self.n_features_, n_docs, self.min_df)
        return self

    # ------------------------------------------------------------------
    def transform(self, dicts: Iterable[dict[str, float]]) -> sparse.csr_matrix:
        if not self.vocabulary_:
            raise RuntimeError("goi fit() truoc")

        indptr = [0]
        indices: list[int] = []
        data: list[float] = []

        norm_slices = {g: self.group_slices_[g] for g in self.normalize_rows
                       if g in self.group_slices_}

        for d in dicts:
            # Chuan hoa G3 theo tong instruction THO (lay tu chinh dict), khong
            # phai theo tong cot con lai sau khi min_df cat - neu khong, ti le
            # se phu thuoc vao vocabulary.
            denom = {}
            for g in norm_slices:
                if g == "G3":
                    total = d.get(G.key(G.G3, "n_instructions"), 0.0)
                else:
                    total = sum(v for k, v in d.items() if G.group_of(k) == g)
                denom[g] = total if total > 0 else 1.0

            row: dict[int, float] = {}
            for k, v in d.items():
                j = self.vocabulary_.get(k)
                if j is None or v == 0:
                    continue
                g = G.group_of(k)
                if g in norm_slices and k.split(G.SEP, 1)[1].startswith(("op:", "op2:")):
                    val = v / denom[g]
                elif self.log1p and g not in norm_slices and v > 0:
                    val = float(np.log1p(v))
                else:
                    val = float(v)
                row[j] = val

            for j in sorted(row):
                indices.append(j)
                data.append(row[j])
            indptr.append(len(indices))

        X = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32),
             np.asarray(indices, dtype=np.int32),
             np.asarray(indptr, dtype=np.int64)),
            shape=(len(indptr) - 1, self.n_features_))
        X.sort_indices()
        return X

    def fit_transform(self, dicts: Iterable[dict[str, float]]) -> sparse.csr_matrix:
        dicts = list(dicts)
        return self.fit(dicts).transform(dicts)

    # ------------------------------------------------------------------
    def group_columns(self, group: str) -> np.ndarray:
        if group not in self.group_slices_:
            return np.array([], dtype=int)
        a, b = self.group_slices_[group]
        return np.arange(a, b)

    def subset(self, X: sparse.csr_matrix, groups: Sequence[str]) -> sparse.csr_matrix:
        """Cat ra chi cac cot cua cac nhom cho truoc - dung cho Bang B."""
        cols = np.concatenate([self.group_columns(g) for g in groups]) \
            if groups else np.array([], dtype=int)
        if cols.size == 0:
            raise ValueError(f"khong co cot nao cho nhom {groups}")
        return X[:, cols]

    def available_groups(self) -> list[str]:
        return [g for g in self.groups if self.group_slices_.get(g, (0, 0))[1]
                > self.group_slices_.get(g, (0, 0))[0]]

    # ------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        log.info("Luu vectorizer -> %s", path)
        return path

    @staticmethod
    def load(path: Path) -> "GroupVectorizer":
        import joblib
        return joblib.load(path)


# --------------------------------------------------------------------------
# Doc feature tu parquet
# --------------------------------------------------------------------------
def load_feature_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "sha256" not in df.columns:
        raise ValueError(f"{path} thieu cot sha256")
    return df


def parse_dicts(df: pd.DataFrame) -> list[dict[str, float]]:
    return [json.loads(s) for s in df["features_json"]]


def load_xy(path: Path, vectorizer: GroupVectorizer | None = None,
            sha_filter: Iterable[str] | None = None
            ) -> tuple[sparse.csr_matrix | None, np.ndarray, pd.DataFrame]:
    """Doc parquet -> (X, y, meta). X=None neu chua co vectorizer (dang fit)."""
    df = load_feature_frame(path)
    if sha_filter is not None:
        keep = set(sha_filter)
        df = df[df["sha256"].isin(keep)].reset_index(drop=True)
    dicts = parse_dicts(df)
    y = df["label"].to_numpy(dtype=int) if "label" in df.columns else np.array([])
    X = vectorizer.transform(dicts) if vectorizer is not None else None
    return X, y, df


def default_vectorizer_path() -> Path:
    return config.MODELS_DIR / "vectorizer.joblib"
