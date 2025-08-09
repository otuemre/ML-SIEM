# src/components/model_trainer.py
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict

import json
import math
import dask.dataframe as dd
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ModelTrainerConfig:
    processed_data_dir: Path = Path("artifacts/data/processed")
    models_dir: Path = Path("artifacts/models")
    metrics_dir: Path = Path("artifacts/metrics")

    model_path: Path = Path("artifacts/models/autoencoder.h5")
    threshold_path: Path = Path("artifacts/models/threshold.json")
    metrics_path: Path = Path("artifacts/metrics/metrics.json")

    # training params
    batch_size: int = 256
    epochs: int = 50
    patience: int = 6
    lr_patience: int = 3
    lr_factor: float = 0.5
    random_state: int = 42

    # threshold percentile on val normals
    threshold_percentile: float = 95.0

class ModelTrainer:
    def __init__(self, config: Optional[ModelTrainerConfig] = None):
        self.config = config or ModelTrainerConfig()
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        self.config.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.feature_cols: List[str] = []
        self.target_col: str = "is_account_takeover"

    # -------------------- Data access --------------------
    def _load_split(self, split: str) -> dd.DataFrame:
        """
        Load one split from processed parquet (partitioned by 'split').
        """
        try:
            path = self.config.processed_data_dir.as_posix()
            logger.info("Loading split=%s from %s", split, path)
            ddf = dd.read_parquet(path, filters=[("split", "==", split)])
            # infer feature columns on first load
            if not self.feature_cols:
                cols = list(ddf.columns)
                cols.remove(self.target_col)
                cols.remove("split")
                self.feature_cols = cols
                logger.info("Feature columns: %s (d=%d)", self.feature_cols, len(self.feature_cols))
            return ddf
        except Exception as e:
            raise CustomException(f"Failed to load split '{split}'", cause=e) from e

    def _partitions(self, ddf: dd.DataFrame) -> Iterable[pd.DataFrame]:
        """
        Yield pandas DataFrames for each Dask partition.
        """
        for part in ddf.to_delayed():
            yield part.compute()

    # -------------------- tf.data generators --------------------
    def _dataset_from_ddf(self, ddf: dd.DataFrame, batch_size: int, include_targets: bool) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset that streams batches from a Dask DataFrame.
        """

        input_dim = len(self.feature_cols)

        def gen():
            for pdf in self._partitions(ddf):
                X = pdf[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
                if include_targets:
                    y = pdf[self.target_col].to_numpy(dtype=np.int8, copy=False)
                # chunk
                for i in range(0, len(X), batch_size):
                    xb = X[i:i + batch_size]
                    if include_targets:
                        yb = y[i:i + batch_size]
                        yield xb, yb
                    else:
                        yield xb

        if include_targets:
            ds = tf.data.Dataset.from_generator(
                gen,
                output_signature=(
                    tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,), dtype=tf.int8),
                ),
            )
            # for AE training, labels = inputs (on normals)
            ds = ds.map(lambda x, y: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = tf.data.Dataset.from_generator(
                gen,
                output_signature=tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
            )

        return ds.prefetch(tf.data.AUTOTUNE)

    # -------------------- Model --------------------
    def _build_autoencoder(self, input_dim: int) -> tf.keras.Model:
        inp = Input(shape=(input_dim,))
        x = Dense(128, activation='relu')(inp)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(x)
        bottleneck = Dense(16, activation='relu')(x)

        x = Dense(32, activation='relu')(bottleneck)
        x = Dense(64, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        out = Dense(input_dim, activation='sigmoid')(x)

        model = Model(inp, out)
        model.compile(optimizer='adam', loss='mse')
        logger.info("Autoencoder built with input_dim=%d, params=%d", input_dim, model.count_params())
        return model

    # -------------------- Thresholding / metrics --------------------
    @staticmethod
    def _mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.mean((a - b) ** 2, axis=1)

    @staticmethod
    def _confusion_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    @staticmethod
    def _prf(stats: Dict[str, int]) -> Dict[str, float]:
        tp, tn, fp, fn = stats["tp"], stats["tn"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    # -------------------- Train / Eval --------------------
    def train(self) -> None:
        try:
            # Load splits
            ddf_train = self._load_split("train")
            ddf_val   = self._load_split("val")
            ddf_test  = self._load_split("test")

            input_dim = len(self.feature_cols)

            # Build datasets
            train_ds = self._dataset_from_ddf(ddf_train, self.config.batch_size, include_targets=True)
            val_ds_inputs = self._dataset_from_ddf(ddf_val, self.config.batch_size, include_targets=True)

            # Keras expects (x,y) for validation. Our val_ds_inputs yields (x, ytrue).
            # Map to (x, x) for val too (we'll recompute errors separately for threshold).
            val_ds = val_ds_inputs.map(lambda x, y: (x, x), num_parallel_calls=tf.data.AUTOTUNE)

            # Build & train model
            model = self._build_autoencoder(input_dim)
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=self.config.patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=self.config.lr_factor, patience=self.config.lr_patience),
            ]

            # We let Keras run to dataset exhaustion per epoch (no steps_per_epoch)
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=1,
            )

            # Save model
            model.save(self.config.model_path)
            logger.info("Saved model to %s", self.config.model_path)

            # Compute validation MSE distribution (normals only) to pick threshold
            val_errors: List[float] = []
            for pdf in self._partitions(ddf_val):
                X = pdf[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
                Xp = model.predict(X, verbose=0)
                errs = self._mse(X, Xp)
                val_errors.append(errs)
            val_errors = np.concatenate(val_errors) if val_errors else np.array([], dtype=np.float32)

            if val_errors.size == 0:
                raise CustomException("No validation errors computed; check val split.")

            threshold = float(np.percentile(val_errors, self.config.threshold_percentile))
            with open(self.config.threshold_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "threshold": threshold,
                        "percentile": self.config.threshold_percentile,
                        "note": "computed on validation normals",
                    },
                    f,
                    indent=2,
                )
            logger.info("Saved threshold=%.6f to %s", threshold, self.config.threshold_path)

            # Evaluate on test (mixed)
            stats_accum = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
            for pdf in self._partitions(ddf_test):
                X = pdf[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
                y = pdf[self.target_col].to_numpy(dtype=np.int8, copy=False)
                Xp = model.predict(X, verbose=0)
                errs = self._mse(X, Xp)
                y_hat = (errs >= threshold).astype(np.int8)  # anomaly if error >= threshold
                s = self._confusion_stats(y, y_hat)
                for k in stats_accum:
                    stats_accum[k] += s[k]

            prf = self._prf(stats_accum)
            metrics = {
                "threshold": threshold,
                "confusion": stats_accum,
                "precision": prf["precision"],
                "recall": prf["recall"],
                "f1": prf["f1"],
            }
            with open(self.config.metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logger.info("Saved metrics to %s | %s", self.config.metrics_path, metrics)

        except Exception as e:
            raise CustomException("Model training failed", cause=e) from e
