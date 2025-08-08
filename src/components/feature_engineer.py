import sys

import dask.dataframe as dd
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class FeatureEngineerConfig:
    keep_dask = False

class FeatureEngineer:
    def __init__(self):
        self.feature_engineer_config = FeatureEngineerConfig()

    @staticmethod
    def extract_ip_octets(df):
        try:
            logging.info('Extracting Octets of IP Address')

            # Divide IP into 4 Octets
            octets = df['IP Address'].str.split('.', n=3, expand=True)

            for i in range(4):
                # Scale Each Octet
                df[f'ip_{i + 1}'] = octets[i].astype(float) / 255.0

            return df

        except Exception as e:
            logging.critical(str(e))
            raise CustomException(e, sys)

    @staticmethod
    def extract_hour_and_day_of_week(df):
        try:
            logging.info('Extracting Hour and Day of the Week')

            # Turn Time to DateTime Object
            df['Login Timestamp'] = dd.to_datetime(df['Login Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

            # Extract the Hour and the Day of the Week
            df['login_hour'] = df['Login Timestamp'].dt.hour
            df['login_day'] = df['Login Timestamp'].dt.weekday

            return df

        except Exception as e:
            logging.critical(str(e))
            raise CustomException(e, sys)

    @staticmethod
    def extract_browser_name(df):
        try:
            logging.info('Extracting Browser Name')

            # Extract the Browser Name using Regex
            df['browser_name'] = df['Browser Name and Version'].str.extract(r'^([^\d]*\d*\s?[A-Za-z]+)', expand=False)

            return df

        except Exception as e:
            logging.critical(str(e))
            raise CustomException(e, sys)

    @staticmethod
    def extract_os_name(df):
        try:
            logging.info('Extracting OS Name')

            # Extract the OS Name using Regex
            df['os_name'] = df['OS Name and Version'].str.extract(r'^(.*?)(?:\s+\d+.*)?$', expand=False).str.strip()

            return df

        except Exception as e:
            logging.critical(str(e))
            raise CustomException(e, sys)

    @staticmethod
    def drop_columns(df):
        try:
            logging.info('Dropping Columns')

            # Column Names to Drop
            columns_to_drop = [
                'index', 'Round-Trip Time [ms]', 'Region', 'City', 'User Agent String', 'ASN', 'User ID',
                'Login Timestamp',
                'Browser Name and Version', 'OS Name and Version', 'IP Address'
            ]

            # Drop Columns
            df = df.drop(columns=columns_to_drop)

            return df

        except Exception as e:
            logging.critical(str(e))
            raise CustomException(e, sys)

    @staticmethod
    def rename_columns(df):
        try:
            logging.info('Renaming Columns')

            # Renaming Dictionary
            rename_dict = {
                "Country": "country_code",
                "Device Type": "device_type",
                "Login Successful": "is_login_success",
                "Is Attack IP": "is_attack_ip",
                "Is Account Takeover": "is_account_takeover",
            }

            # Rename columns
            df = df.rename(columns=rename_dict)

            return df

        except Exception as e:
            logging.critical(str(e))
            raise CustomException(e, sys)

    @staticmethod
    def handle_null(df):
        try:
            logging.info('Dropping NaN Values')

            # Calculate the Size before and after NaNs
            original_len = len(df)
            df = df.dropna()
            new_len = len(df)

            # Output the result
            logging.info(f"Dropped {original_len - new_len} NaNs ({(original_len - new_len) / original_len:.2%})")

            return df

        except Exception as e:
            logging.critical(str(e))
            raise CustomException(e, sys)

    @staticmethod
    def handle_duplicates(df):
        try:
            logging.info('Dropping Duplicated Values')

            # Calculate the Size before and after duplicates
            original_len = len(df)
            df = df.drop_duplicates()
            new_len = len(df)

            # Output the result
            logging.info(f"Dropped {original_len - new_len} duplicates ({(original_len - new_len) / original_len:.2%})")

            return df

        except Exception as e:
            logging.critical(str(e))
            raise CustomException(e, sys)

    def initiate_feature_engineer(self, df):
        try:
            logging.info('Starting to Preprocess Dataset')

            # Preprocess the Dataset
            df = self.extract_ip_octets(df)
            df = self.extract_hour_and_day_of_week(df)
            df = self.extract_browser_name(df)
            df = self.extract_os_name(df)
            df = self.drop_columns(df)
            df = self.rename_columns(df)
            df = self.handle_null(df)

            # Compute Dask DataFrame to apply preprocessing
            df = df.compute()

            # If keep dask
            if self.feature_engineer_config.keep_dask and isinstance(df, pd.DataFrame):
                df = dd.from_pandas(df)

            return df

        except Exception as e:
            logging.critical(str(e))
            raise CustomException(e, sys)