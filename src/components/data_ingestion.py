import os
import sys
import dask.dataframe as dd

from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion Method/Component...")

        try:
            df = dd.read_csv('data/rba-dataset.csv')
            logging.info("The dataset has been loaded...")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, single_file=True)

            logging.info("Train-Test split has been initiated...")
            normal_data = df[df['Is Account Takeover'] == 0]
            anomalous_data = df[df['Is Account Takeover'] == 1]

            train_set = normal_data.sample(frac=0.8, random_state=42)
            remaining_normal = dd.concat([normal_data, train_set]).drop_duplicates()
            test_set = dd.concat([remaining_normal, anomalous_data])

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, single_file=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, single_file=True)
            logging.info("Ingestion of data is completed...")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.critical(str(e))
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()