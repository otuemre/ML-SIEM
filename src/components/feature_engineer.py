# src/components/feature_engineer.py
from dataclasses import dataclass
from pathlib import Path
import dask.dataframe as dd
import pyarrow as pa

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

@dataclass
class FeatureEngineerConfig:
    raw_data_dir: Path = Path("artifacts/data/raw")
    fe_data_dir: Path = Path("artifacts/data/fe")

class FeatureEngineer:
    def __init__(self, config: FeatureEngineerConfig | None = None):
        self.config = config or FeatureEngineerConfig()
        self.config.fe_data_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- helpers -----------------
    @staticmethod
    def _extract_ip_octets(df: dd.DataFrame) -> dd.DataFrame:
        """Create ip_1..ip_4 scaled (0..1) from ip_address."""
        logger.info("Extracting IP octets -> ip_1..ip_4")
        if "ip_address" not in df.columns:
            raise CustomException("Column 'ip_address' not found for IP octet extraction")

        octets = df["ip_address"].str.split(".", n=3, expand=True)
        for i in range(4):
            df[f"ip_{i+1}"] = dd.to_numeric(octets[i], errors="coerce") / 255.0
        return df

    @staticmethod
    def _extract_hour_and_day_of_week(df: dd.DataFrame) -> dd.DataFrame:
        """Create login_hour (0-23) and login_day (0-6) from login_timestamp."""
        logger.info("Extracting login_hour and login_day from login_timestamp")
        if "login_timestamp" not in df.columns:
            raise CustomException("Column 'login_timestamp' not found for time feature extraction")

        dt = dd.to_datetime(df["login_timestamp"], errors="coerce")
        df["login_hour"] = dt.dt.hour.astype("uint8")
        df["login_day"]  = dt.dt.weekday.astype("uint8")
        return df

    @staticmethod
    def _extract_browser_name(df: dd.DataFrame) -> dd.DataFrame:
        """Extract browser_name from browser_name_and_version."""
        logger.info("Extracting browser_name")
        col = "browser_name_and_version"
        if col not in df.columns:
            logger.info("Column '%s' not present, skipping browser_name extraction", col)
            return df

        df["browser_name"] = df[col].str.extract(r"^([A-Za-z]+)", expand=False)
        return df

    @staticmethod
    def _extract_os_name(df: dd.DataFrame) -> dd.DataFrame:
        """Extract os_name from os_name_and_version."""
        logger.info("Extracting os_name")
        col = "os_name_and_version"
        if col not in df.columns:
            logger.info("Column '%s' not present, skipping os_name extraction", col)
            return df

        df["os_name"] = df[col].str.extract(r"^([^\d]+)", expand=False).str.strip()
        return df

    @staticmethod
    def _rename_columns(df: dd.DataFrame) -> dd.DataFrame:
        """Normalize semantic names after ingestion normalization."""
        logger.info("Renaming columns to canonical feature names")
        rename_map = {
            "country": "country_code",
            "login_successful": "is_login_success",
        }
        existing = {k: v for k, v in rename_map.items() if k in df.columns}
        if existing:
            df = df.rename(columns=existing)
        return df

    @staticmethod
    def _handle_nulls(df: dd.DataFrame) -> dd.DataFrame:
        """
        Fill obvious nulls instead of dropping 95% of the dataset.
        - browser_name / os_name -> 'Unknown'
        - boolean flags -> False (conservative)
        """
        logger.info("Filling nulls for key columns")
        if "browser_name" in df.columns:
            df["browser_name"] = df["browser_name"].fillna("Unknown")
        if "os_name" in df.columns:
            df["os_name"] = df["os_name"].fillna("Unknown")

        for b in ["is_login_success", "is_attack_ip", "is_account_takeover"]:
            if b in df.columns:
                # Cast to bool after fill
                df[b] = df[b].fillna(False).astype("bool")

        # If any IP octet is missing, set to 0.0 (unknown/invalid IP parse)
        for i in range(1, 5):
            col = f"ip_{i}"
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # login_hour/day: set unknowns to 0 (midnight / Monday) rather than dropping
        for c in ["login_hour", "login_day"]:
            if c in df.columns:
                df[c] = df[c].fillna(0)

        return df

    @staticmethod
    def _drop_columns(df: dd.DataFrame) -> dd.DataFrame:
        """Drop heavy/high-cardinality columns not needed after FE."""
        logger.info("Dropping high-cardinality/unused columns")
        drop_cols = [
            "index",
            "round-trip_time_[ms]",
            "region",
            "city",
            "user_agent_string",
            "asn",
            "user_id",
            "login_timestamp",
            "browser_name_and_version",
            "os_name_and_version",
            "ip_address",
        ]
        present = [c for c in drop_cols if c in df.columns]
        if present:
            df = df.drop(columns=present)
        return df

    @staticmethod
    def _coerce_dtypes(df: dd.DataFrame) -> dd.DataFrame:
        """Force stable dtypes across partitions before writing."""
        logger.info("Coercing dtypes before Parquet write")
        dtype_map = {
            "country_code": "string",
            "device_type": "string",
            "browser_name": "string",
            "os_name": "string",

            "is_login_success": "bool",
            "is_attack_ip": "bool",
            "is_account_takeover": "bool",

            "ip_1": "float64",
            "ip_2": "float64",
            "ip_3": "float64",
            "ip_4": "float64",

            "login_hour": "uint8",
            "login_day": "uint8",
        }
        existing = {k: v for k, v in dtype_map.items() if k in df.columns}
        return df.astype(existing)

    @staticmethod
    def _pyarrow_schema() -> pa.schema:
        """Explicit schema to keep pyarrow from guessing poorly."""
        return pa.schema([
            pa.field("country_code", pa.string()),
            pa.field("device_type", pa.string()),
            pa.field("is_login_success", pa.bool_()),
            pa.field("is_attack_ip", pa.bool_()),
            pa.field("is_account_takeover", pa.bool_()),
            pa.field("ip_1", pa.float64()),
            pa.field("ip_2", pa.float64()),
            pa.field("ip_3", pa.float64()),
            pa.field("ip_4", pa.float64()),
            pa.field("login_hour", pa.uint8()),
            pa.field("login_day", pa.uint8()),
            pa.field("browser_name", pa.string()),
            pa.field("os_name", pa.string()),
        ])

    # ----------------- public API -----------------
    def run(self, source_dir: str | Path | None = None) -> Path:
        """
        Load raw dataset (Parquet) from artifacts/data/raw, engineer features,
        and write Parquet dataset to artifacts/data/fe. Returns output dir path.
        """
        try:
            raw_dir = Path(source_dir) if source_dir else self.config.raw_data_dir
            out_dir = self.config.fe_data_dir

            logger.info("Loading raw dataset from %s", raw_dir)
            df = dd.read_parquet(raw_dir.as_posix())

            # Feature steps
            df = self._extract_ip_octets(df)
            df = self._extract_hour_and_day_of_week(df)
            df = self._extract_browser_name(df)
            df = self._extract_os_name(df)
            df = self._rename_columns(df)
            df = self._handle_nulls(df)
            df = self._drop_columns(df)

            # Coerce dtypes so all partitions match
            df = self._coerce_dtypes(df)

            # Clean output dir to avoid mixing partitions during dev
            if out_dir.exists():
                for p in out_dir.glob("*"):
                    if p.is_file():
                        p.unlink()
                    else:
                        for q in p.rglob("*"):
                            if q.is_file():
                                q.unlink()
                        p.rmdir()

            # Write with explicit schema
            df.to_parquet(
                out_dir.as_posix(),
                write_index=False,
                engine="pyarrow",
                schema=self._pyarrow_schema(),
            )
            logger.info("Wrote feature-engineered dataset to %s", out_dir)
            return out_dir

        except Exception as e:
            logger.critical("Feature engineering failed: %s", e)
            raise CustomException("Feature engineering failed", cause=e) from e
