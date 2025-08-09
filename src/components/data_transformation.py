from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import dask.dataframe as dd
import numpy as np
import pandas as pd
import json
import joblib

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DataTransformationConfig:
    fe_data_dir: Path = Path("artifacts/data/fe")
    processed_data_dir: Path = Path("artifacts/data/processed")
    models_dir: Path = Path("artifacts/models")
    category_maps_path: Path = Path("artifacts/models/category_maps.pkl")
    minmax_stats_path: Path = Path("artifacts/models/minmax_stats.json")
    train_frac: float = 0.8
    val_frac: float = 0.1
    random_state: int = 42

class DataTransformation:
    def __init__(self, config: Optional[DataTransformationConfig] = None):
        self.config = config or DataTransformationConfig()
        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        # Columns (target excluded from feature lists)
        self.categorical_cols: List[str] = ["country_code", "device_type", "browser_name", "os_name"]
        self.boolean_feature_cols: List[str] = ["is_login_success", "is_attack_ip"]   # target excluded!
        self.numeric_cols: List[str] = ["ip_1", "ip_2", "ip_3", "ip_4", "login_hour", "login_day"]
        self.target_col: str = "is_account_takeover"

        self.category_maps: Dict[str, Dict[str, int]] = {}
        self.minmax_stats: Dict[str, Dict[str, float]] = {}

    # --------------- helpers ---------------
    def _load_ddf(self) -> dd.DataFrame:
        try:
            logger.info("Loading FE dataset from %s", self.config.fe_data_dir)
            ddf = dd.read_parquet(self.config.fe_data_dir.as_posix())
            required = set(self.categorical_cols + self.boolean_feature_cols + self.numeric_cols + [self.target_col])
            missing = [c for c in required if c not in ddf.columns]
            if missing:
                raise CustomException(f"Missing columns in FE dataset: {missing}")
            ddf = ddf[list(required)]
            logger.info("FE dtypes: %s", {c: str(t) for c, t in ddf.dtypes.items()})
            return ddf
        except Exception as e:
            raise CustomException("Failed to load FE dataset", cause=e) from e

    def _split_ddf(self, ddf: dd.DataFrame):
        try:
            normals = ddf[ddf[self.target_col] == 0]
            anomalies = ddf[ddf[self.target_col] == 1]

            n_norm = int(normals.shape[0].compute())
            n_anom = int(anomalies.shape[0].compute())
            logger.info("Normals: %d | Anomalies: %d", n_norm, n_anom)

            train_normals, remain_normals = normals.random_split(
                [self.config.train_frac, 1.0 - self.config.train_frac],
                random_state=self.config.random_state,
            )
            rel = (self.config.val_frac / (1.0 - self.config.train_frac)) if (1.0 - self.config.train_frac) > 0 else 0.0
            val_normals, test_normals = remain_normals.random_split(
                [rel, 1.0 - rel],
                random_state=self.config.random_state,
            )
            test = dd.concat([test_normals, anomalies], axis=0)

            logger.info(
                "Split sizes (~) -> train_normals: %s | val_normals: %s | test(mixed): %s",
                int(train_normals.shape[0].compute()),
                int(val_normals.shape[0].compute()),
                int(test.shape[0].compute()),
            )
            return train_normals, val_normals, test
        except Exception as e:
            raise CustomException("Failed to split dataset with Dask", cause=e) from e

    def _build_category_maps(self, train_normals: dd.DataFrame):
        try:
            maps: Dict[str, Dict[str, int]] = {}
            for col in self.categorical_cols:
                cats = (
                    train_normals[col]
                    .astype("string")
                    .dropna()
                    .unique()
                    .compute()
                    .tolist()
                )
                cats = sorted([c for c in cats if c is not None])
                maps[col] = {cat: i for i, cat in enumerate(cats)}
                logger.info("Built category map for %s: %d categories", col, len(maps[col]))
            self.category_maps = maps
            joblib.dump(maps, self.config.category_maps_path)
            logger.info("Saved category maps to %s", self.config.category_maps_path)
        except Exception as e:
            raise CustomException("Failed to build/save category maps", cause=e) from e

    def _encode_categoricals(self, ddf: dd.DataFrame) -> dd.DataFrame:
        try:
            for col, mapping in self.category_maps.items():
                if col not in ddf.columns:
                    continue
                # Provide meta to silence warnings and set dtype
                ddf[col] = (
                    ddf[col]
                    .astype("string")
                    .map(mapping, meta=(col, "float64"))
                    .fillna(-1)
                    .astype("int32")
                )
            return ddf
        except Exception as e:
            raise CustomException("Failed to encode categoricals", cause=e) from e

    def _coerce_booleans(self, ddf: dd.DataFrame) -> dd.DataFrame:
        try:
            for b in self.boolean_feature_cols + [self.target_col]:
                if b in ddf.columns:
                    ddf[b] = ddf[b].astype("bool").astype("int8")
            return ddf
        except Exception as e:
            raise CustomException("Failed to coerce boolean columns", cause=e) from e

    def _compute_minmax(self, train_normals: dd.DataFrame):
        try:
            # Scale: numeric + boolean_features + encoded categoricals
            cols = [c for c in (self.numeric_cols + self.boolean_feature_cols + self.categorical_cols) if c in train_normals.columns]
            mins = train_normals[cols].min().compute()
            maxs = train_normals[cols].max().compute()

            stats = {}
            for c in cols:
                mn = float(mins[c]); mx = float(maxs[c])
                if not np.isfinite(mn): mn = 0.0
                if not np.isfinite(mx): mx = 1.0
                if mx == mn: mx = mn + 1.0
                stats[c] = {"min": mn, "max": mx}
            self.minmax_stats = stats

            with open(self.config.minmax_stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            logger.info("Saved min-max stats to %s", self.config.minmax_stats_path)
        except Exception as e:
            raise CustomException("Failed to compute/save min-max stats", cause=e) from e

    def _apply_minmax(self, ddf: dd.DataFrame) -> dd.DataFrame:
        try:
            for c, mm in self.minmax_stats.items():
                if c not in ddf.columns:
                    continue
                mn = mm["min"]; mx = mm["max"]; rng = (mx - mn) if (mx - mn) != 0 else 1.0
                ddf[c] = (ddf[c].astype("float32") - mn) / rng
            return ddf
        except Exception as e:
            raise CustomException("Failed to apply min-max scaling", cause=e) from e

    def _write_processed(self, ddf: dd.DataFrame):
        try:
            out_dir = self.config.processed_data_dir
            # clean
            if out_dir.exists():
                for p in out_dir.glob("*"):
                    if p.is_file():
                        p.unlink()
                    else:
                        for q in p.rglob("*"):
                            if q.is_file():
                                q.unlink()
                        p.rmdir()

            # split should be string for stable partitioning
            if "split" not in ddf.columns:
                raise CustomException("Internal error: 'split' column missing before write")
            ddf["split"] = ddf["split"].astype("string")

            ddf.to_parquet(
                out_dir.as_posix(),
                write_index=False,
                partition_on=["split"],
                engine="pyarrow",
            )
            logger.info("Wrote processed dataset to %s (partitioned by split)", out_dir)
            return out_dir
        except Exception as e:
            raise CustomException("Failed to write processed dataset", cause=e) from e

    # --------------- public API ---------------
    def fit_transform(self) -> Path:
        try:
            ddf = self._load_ddf()

            # Split
            train_normals, val_normals, test_mixed = self._split_ddf(ddf)
            train_normals = train_normals.assign(split="train")
            val_normals   = val_normals.assign(split="val")
            test_mixed    = test_mixed.assign(split="test")

            # Category maps from TRAIN NORMALS ONLY
            self._build_category_maps(train_normals)

            # Encode categoricals
            train_normals = self._encode_categoricals(train_normals)
            val_normals   = self._encode_categoricals(val_normals)
            test_mixed    = self._encode_categoricals(test_mixed)

            # Booleans -> 0/1 (includes target)
            train_normals = self._coerce_booleans(train_normals)
            val_normals   = self._coerce_booleans(val_normals)
            test_mixed    = self._coerce_booleans(test_mixed)

            # MinMax on TRAIN NORMALS, then apply to all
            self._compute_minmax(train_normals)
            train_normals = self._apply_minmax(train_normals)
            val_normals   = self._apply_minmax(val_normals)
            test_mixed    = self._apply_minmax(test_mixed)

            # Combine & keep only features + target + split (no duplicates)
            processed = dd.concat([train_normals, val_normals, test_mixed], axis=0)

            feature_cols = self.categorical_cols + self.numeric_cols + self.boolean_feature_cols
            keep_cols = feature_cols + [self.target_col, "split"]
            # de-duplicate while preserving order
            seen = set(); keep_cols = [c for c in keep_cols if (c not in seen and not seen.add(c))]
            processed = processed[keep_cols]

            return self._write_processed(processed)

        except Exception as e:
            raise CustomException("Data transformation failed", cause=e) from e

    def transform_only(self, df_or_path) -> pd.DataFrame:
        try:
            if isinstance(df_or_path, (str, Path)):
                ddf = dd.read_parquet(str(df_or_path))
                df = ddf.compute()
            else:
                df = df_or_path.copy()

            if not self.category_maps:
                self.category_maps = joblib.load(self.config.category_maps_path)
            if not self.minmax_stats:
                with open(self.config.minmax_stats_path, "r", encoding="utf-8") as f:
                    self.minmax_stats = json.load(f)

            # Encode categoricals
            for col, mapping in self.category_maps.items():
                if col in df.columns:
                    df[col] = df[col].astype("string").map(mapping).fillna(-1).astype("int32")

            # Booleans -> 0/1 (features only)
            for b in self.boolean_feature_cols:
                if b in df.columns:
                    df[b] = df[b].astype("bool").astype("int8")

            # MinMax
            for c, mm in self.minmax_stats.items():
                if c in df.columns:
                    mn = mm["min"]; mx = mm["max"]; rng = (mx - mn) if (mx - mn) != 0 else 1.0
                    df[c] = (df[c].astype("float32") - mn) / rng

            feat_cols = self.categorical_cols + self.numeric_cols + self.boolean_feature_cols
            feat_cols = [c for c in feat_cols if c in df.columns]
            return df[feat_cols]

        except Exception as e:
            raise CustomException("Transform-only failed", cause=e) from e
