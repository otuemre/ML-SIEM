from dataclasses import dataclass
from pathlib import Path
import dask.dataframe as dd

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DataIngestionConfig:
    # Write as a Dask Parquet dataset (directory)
    raw_data_dir: Path = Path("artifacts/data/raw")

class DataIngestion:
    def __init__(self, config: DataIngestionConfig | None = None):
        self.config = config or DataIngestionConfig()
        self.config.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_columns(self, df: dd.DataFrame) -> dd.DataFrame:
        """
        Lowercase + snake_case the columns and fix known header variants.
        """
        # snake-case all columns
        rename_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=rename_map)

        return df

    def run(self, source_path: str = "data/rba-dataset.csv") -> Path:
        """
        Read raw dataset (CSV or Parquet), normalize headers, persist as Parquet.
        Returns the directory path of the Parquet dataset.
        """
        try:
            logger.info("Starting data ingestion from %s", source_path)

            if source_path.endswith(".parquet"):
                df = dd.read_parquet(source_path)
            else:
                df = dd.read_csv(source_path, assume_missing=True, blocksize="64MB")

            logger.info("Loaded raw dataset with %d columns", len(df.columns))

            df = self._normalize_columns(df)

            # Quick sanity logs
            row_count = df.shape[0].compute()
            logger.info("Normalized columns: %s", list(df.columns))
            logger.info("Row count (approx): %s", row_count)

            # Persist as Parquet dataset (partitioned). Overwrite safely.
            out_dir = self.config.raw_data_dir
            if out_dir.exists():
                # avoid mixing old/new partitions during dev
                for p in out_dir.glob("*"):
                    if p.is_file():
                        p.unlink()
                    else:
                        # remove partition dirs
                        for q in p.rglob("*"):
                            if q.is_file():
                                q.unlink()
                        p.rmdir()

            df.to_parquet(out_dir.as_posix(), write_index=False)
            logger.info("Wrote raw Parquet dataset to %s", out_dir)

            return out_dir

        except Exception as e:
            logger.critical("Data ingestion failed: %s", e)
            raise CustomException("Data ingestion failed", cause=e) from e

if __name__ == '__main__':
    data_ingestion = DataIngestion()
    output_dir = data_ingestion.run()