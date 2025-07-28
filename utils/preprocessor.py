# Imports
import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import joblib
import logging
import os

class Preprocessor:
    def __init__(self, data_path, data_output, model_output, log_output):
        self.df = None
        self.data_path = data_path
        self.data_output = data_output
        self.model_output = model_output
        self.encoders = None
        self.scaler = None

        # Logging Config
        logging.basicConfig(
            filename=log_output,
            encoding='utf-8',
            filemode='a',
            format='{asctime} - {levelname} - {message}',
            style='{',
            datefmt='%d/%m/%Y - %H:%M',
            level=logging.DEBUG
        )

    def __load_data(self, data_path):
        print('[INFO] - Loading the Dataset...')
        logging.info('Loading the Dataset...')

        try:
            # Load Dataset
            self.df = dd.read_csv(data_path)
        except FileNotFoundError:
            print(f"[CRITICAL] - File NOT exists, please check the path:", data_path)
            logging.critical(f'File NOT exists, please check the path: {data_path}')
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

    def __extract_ip_octets(self):
        print('[INFO] - Extracting Octets of IP Address...')
        logging.info('Extracting Octets of IP Address...')

        # Divide IP into 4 Octets
        octets = self.df['IP Address'].str.split('.', n=3, expand=True)

        for i in range(4):
            # Scale Each Octet
            self.df[f'ip_{i+1}'] = octets[i].astype(float) / 255.0

    def __extract_hour_and_day_of_week(self):
        print('[INFO] - Extracting Hour and Day of the Week...')
        logging.info('Extracting Hour and Day of the Week...')

        # Turn Time to DateTime Object
        self.df['Login Timestamp'] = dd.to_datetime(self.df['Login Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

        # Extract the Hour and the Day of the Week
        self.df['login_hour'] = self.df['Login Timestamp'].dt.hour
        self.df['login_day'] = self.df['Login Timestamp'].dt.weekday

    def __extract_browser_name(self):
        print('[INFO] - Extracting Browser Name...')
        logging.info('Extracting Browser Name...')

        # Extract the Browser Name using Regex
        self.df['browser_name'] = self.df['Browser Name and Version'].str.extract(r'^([^\d]*\d*\s?[A-Za-z]+)', expand=False)

    def __extract_os_name(self):
        print('[INFO] - Extracting OS Name...')
        logging.info('Extracting OS Name...')

        # Extract the OS Name using Regex
        self.df['os_name'] = self.df['OS Name and Version'].str.extract(r'^(.*?)(?:\s+\d+.*)?$', expand=False).str.strip()

    def __drop_columns(self):
        print('[INFO] - Dropping Columns...')
        logging.info('Dropping Columns...')

        # Column Names to Drop
        columns_to_drop = [
            'index', 'Round-Trip Time [ms]', 'Region', 'City', 'User Agent String', 'ASN', 'User ID', 'Login Timestamp',
            'Browser Name and Version', 'OS Name and Version', 'IP Address'
        ]

        # Drop Columns
        self.df = self.df.drop(columns=columns_to_drop)

    def __rename_columns(self):
        print('[INFO] - Renaming Columns...')
        logging.info('Renaming Columns...')

        # Renaming Dictionary
        rename_dict = {
            "Country": "country_code",
            "Device Type": "device_type",
            "Login Successful": "is_login_success",
            "Is Attack IP": "is_attack_ip",
            "Is Account Takeover": "is_account_takeover",
        }

        # Rename columns
        self.df = self.df.rename(columns=rename_dict)

    def __handle_null(self):
        print('[INFO] - Dropping NaN Values...')
        logging.info('Dropping NaN Values...')

        # Calculate the Size before and after NaNs
        original_len = len(self.df)
        self.df = self.df.dropna()
        new_len = len(self.df)

        # Output the result
        print(f"Dropped {original_len - new_len} NaNs ({(original_len - new_len) / original_len:.2%})")

    @staticmethod
    def handle_duplicates(dataframe):
        print('[INFO] - Dropping Duplicated Values...')
        logging.info('Dropping Duplicated Values...')

        # Calculate the Size before and after duplicates
        original_len = len(dataframe)
        dataframe = dataframe.drop_duplicates()
        new_len = len(dataframe)

        # Output the result
        print(f"Dropped {original_len - new_len} duplicates ({(original_len - new_len) / original_len:.2%})")

        return dataframe

    def process_features(self, output=False):
        print('[INFO] - Starting to Preprocess Dataset...')
        logging.info('Starting to Preprocess Dataset...')

        # Load the Dataset
        self.__load_data(self.data_path)

        # Preprocess the Dataset
        self.__extract_ip_octets()
        self.__extract_hour_and_day_of_week()
        self.__extract_browser_name()
        self.__extract_os_name()
        self.__drop_columns()
        self.__rename_columns()
        self.__handle_null()

        # Compute Dask DataFrame to apply preprocessing
        self.df = self.df.compute()

        # Return condition
        if output:
            return self.df

        return None

    def encode_train(self, output=False):
        print("[INFO] - Encoding Columns...")
        logging.info('Encoding Columns...')

        # Check if Dask DataFrame
        if isinstance(self.df, dd.DataFrame):
            self.df = self.df.compute()

        # Columns to Encode
        cols_to_encode = ['country_code', 'device_type', 'browser_name', 'os_name']

        # Create Encoders Dictionary
        self.encoders = {}

        # Encode String Columns
        for col in cols_to_encode:
            label_encoder = LabelEncoder()
            self.df[col] = label_encoder.fit_transform(self.df[col].astype(str))
            self.encoders[col] = label_encoder

        # Encode Boolean Columns
        bool_cols = ['is_login_success', 'is_attack_ip', 'is_account_takeover']
        for col in bool_cols:
            self.df[col] = self.df[col].astype(int)

        # Return condition
        if output:
            return self.df, self.encoders

        return None

    def scale_train(self, output=False):
        print("[INFO] - Scaling Columns...")
        logging.info('Scaling Columns...')

        # Check if Dask DataFrame
        if isinstance(self.df, dd.DataFrame):
            self.df = self.df.compute()

        # Columns to Scale
        cols_to_scale = [
            'country_code', 'device_type', 'is_login_success', 'is_attack_ip',
            'login_hour', 'login_day', 'browser_name', 'os_name'
        ]

        # Create MinMax Scaler
        self.scaler = MinMaxScaler()

        # Scale the Columns
        self.df[cols_to_scale] = self.scaler.fit_transform(self.df[cols_to_scale])

        # Return condition
        if output:
            return self.df, self.encoders

        return None

    def cast_types(self, force_dask, output=False):
        print("[INFO] - Casting Data Types...")
        logging.info('Casting Data Types...')

        # Check if Pandas DataFrame
        if isinstance(self.df, pd.DataFrame) and force_dask:
            # Make the df Dask DataFrame
            self.df = dd.from_pandas(self.df)

        # Type Casting
        self.df['country_code'] = self.df['country_code'].astype(np.float32)
        self.df['device_type'] = self.df['device_type'].astype(np.float32)
        self.df['is_login_success'] = self.df['is_login_success'].astype(np.float32)
        self.df['is_attack_ip'] = self.df['is_attack_ip'].astype(np.float32)
        self.df['login_hour'] = self.df['login_hour'].astype(np.float32)
        self.df['login_day'] = self.df['login_day'].astype(np.float32)
        self.df['browser_name'] = self.df['browser_name'].astype(np.float32)
        self.df['os_name'] = self.df['os_name'].astype(np.float32)
        self.df['ip_1'] = self.df['ip_1'].astype(np.float32)
        self.df['ip_2'] = self.df['ip_2'].astype(np.float32)
        self.df['ip_3'] = self.df['ip_3'].astype(np.float32)
        self.df['ip_4'] = self.df['ip_4'].astype(np.float32)

        # Return condition
        if output:
            return self.df

        return None

    def save_dataset(self):
        print(f'[INFO] - Saving Dataset to {self.data_output}/preprocessed.csv')
        logging.info(f'Saving Dataset to {self.data_output}/preprocessed.csv')

        # Create the Full Path
        full_path = os.path.join(self.data_output, 'preprocessed.csv')

        # Create Folder if NOT Exists
        # os.makedirs(self.data_path, exist_ok=True)

        # Save Dataset
        self.df.to_csv(full_path, index=False, single_file=True)

        print(f'[SUCCESS] - Dataset Saved to {self.data_output}/preprocessed.csv')
        logging.info(f'Dataset Saved to {self.data_output}/preprocessed.csv')

    def save_encoders(self):
        print(f'[INFO] - Saving Encoders to {self.model_output}/label_encoders.pkl')
        logging.info(f'Saving Encoders to {self.model_output}/label_encoders.pkl')

        # Create the Full Path
        full_path = os.path.join(self.model_output, 'label_encoders.pkl')

        # Create Folder if NOT Exists
        # os.makedirs(self.model_output, exist_ok=True)

        # Save the Encoders
        joblib.dump(self.encoders, full_path)

        print(f'[SUCCESS] - Encoders Saved to {self.model_output}/label_encoders.pkl')
        logging.info(f'Encoders Saved to {self.model_output}/label_encoders.pkl')

    def save_scaler(self):
        print(f'[INFO] - Saving Scaler to {self.model_output}/minmax_scaler.pkl')
        logging.info(f'Saving Scaler to {self.model_output}/minmax_scaler.pkl')

        # Create the Full Path
        full_path = os.path.join(self.model_output, 'minmax_scaler.pkl')

        # Create Folder if NOT Exists
        # os.makedirs(self.model_output, exist_ok=True)

        # Save the Encoders
        joblib.dump(self.scaler, full_path)

        print(f'[SUCCESS] - Scaler Saved to {self.model_output}/minmax_scaler.pkl')
        logging.info(f'Scaler Saved to {self.model_output}/minmax_scaler.pkl')

    def get_df(self):
        print('[INFO] - Returning DataFrame...')
        logging.info('Returning DataFrame...')

        # Return the DataFrame
        return self.df

    def get_encoders(self):
        print('[INFO] - Returning Encoders...')
        logging.info('Returning Encoders...')
        # Return the Encoders
        return self.encoders

    def get_scaler(self):
        print('[INFO] - Returning Scaler...')
        logging.info('Returning Scaler...')

        # Return the Scaler
        return self.scaler