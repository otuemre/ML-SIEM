from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.exception import CustomException
from src.logger import get_logger
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.feature_engineer import FeatureEngineer, FeatureEngineerConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

logger = get_logger(__name__)

def setup_gpu_logging():
    """Log whether a GPU is available and enable memory growth if possible."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical = tf.config.list_logical_devices("GPU")
            logger.info("GPU detected: %d physical, %d logical. Using: %s",
                        len(gpus), len(logical), logical[0].name if logical else "N/A")
        except RuntimeError as e:
            logger.error("Could not set GPU memory growth: %s", e)
    else:
        logger.warning("No GPU detected — training will run on CPU.")

def run_pipeline(source_path: str | Path = "data/rba-dataset.csv") -> None:
    """
    Orchestrates the end-to-end training pipeline:
      1) Ingest raw -> artifacts/data/raw
      2) Feature engineer -> artifacts/data/fe
      3) Transform (split + encode + scale) -> artifacts/data/processed
      4) Train model -> artifacts/models + artifacts/metrics
    """
    try:
        t0 = time.time()

        # Seeds for repeatability-ish
        tf.get_logger().setLevel("ERROR")
        tf.random.set_seed(42)
        np.random.seed(42)

        # GPU setup
        setup_gpu_logging()

        # ---- 1) Ingestion
        logger.info("STEP 1/4: Data ingestion")
        ingestion = DataIngestion(DataIngestionConfig())
        raw_dir = ingestion.run(str(source_path))
        logger.info("Ingestion done: %s", raw_dir)

        # ---- 2) Feature Engineering
        logger.info("STEP 2/4: Feature engineering")
        fe = FeatureEngineer(FeatureEngineerConfig(raw_data_dir=raw_dir))
        fe_dir = fe.run()
        logger.info("Feature engineering done: %s", fe_dir)

        # ---- 3) Transformation
        logger.info("STEP 3/4: Data transformation")
        dt_conf = DataTransformationConfig(fe_data_dir=fe_dir)
        transformer = DataTransformation(dt_conf)
        processed_dir = transformer.fit_transform()
        logger.info("Transformation done: %s", processed_dir)

        # ---- 4) Training
        logger.info("STEP 4/4: Model training")
        trainer = ModelTrainer(ModelTrainerConfig(processed_data_dir=processed_dir))
        trainer.train()
        logger.info("Training complete.")

        logger.info("Pipeline finished in %.2f minutes", (time.time() - t0) / 60.0)

    except CustomException as e:
        logger.critical("Pipeline failed with CustomException:\n%s", e)
        raise
    except Exception as e:
        logger.critical("Pipeline failed with unexpected error: %s", e)
        raise CustomException("Train pipeline failed", cause=e) from e

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML-SIEM training pipeline.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/rba-dataset.csv",
        help="Path to raw source data (CSV or Parquet). Default: data/rba-dataset.csv",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.source)
