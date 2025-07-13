# Imports
import pandas as pd
import dask.dataframe as dd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

# Config
DATA_PATH = './data/'
DATA_OUTPUT = './data/train/'
MODEL_DIR = './models/train/'
SCALER_PATH = os.path.join(MODEL_DIR, 'min_max_scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
AUTOENCODER_PATH = os.path.join(MODEL_DIR, 'autoencoder_model')

# Load the Data
def load_data(file_name='rba-dataset.csv'):
    print('[INFO] - Loading the Dataset...')
    file_path = os.path.join(DATA_PATH, file_name)
    df = dd.read_csv(file_path)
    return df

def extract_ip_octets(df):
    print('[INFO] - Extracting Octets of IP Address...')
    octets = df['IP Address'].str.split('.', n=3, expand=True)

    for i in range(4):
        df[f'ip_{i+1}'] = octets[i].astype(float) / 255.0

    return df

def extract_hour_and_day_of_week(df):
    print('[INFO] - Extracting Hour and Day of the Week...')
    df['Login Timestamp'] = dd.to_datetime(df['Login Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

    df['login_hour'] = df['Login Timestamp'].dt.hour
    df['login_day'] = df['Login Timestamp'].dt.weekday

    return df

def extract_browser_name(df):
    print('[INFO] - Extracting Browser Name...')
    df['browser_name'] = df['Browser Name and Version'].str.extract(r'^([^\d]*\d*\s?[A-Za-z]+)', expand=False)

    return df

def extract_os_name(df):
    print('[INFO] - Extracting OS Name...')
    df['os_name'] = df['OS Name and Version'].str.extract(r'^(.*?)(?:\s+\d+.*)?$', expand=False).str.strip()

    return df

def drop_columns(df):
    print('[INFO] - Dropping Columns...')
    columns_to_drop = [
        'index', 'Round-Trip Time [ms]', 'Region', 'City', 'User Agent String', 'ASN', 'User ID', 'Login Timestamp',
        'Browser Name and Version', 'OS Name and Version', 'IP Address'
    ]

    df = df.drop(columns=columns_to_drop)

    return df

def rename_columns(df):
    print('[INFO] - Renaming Columns...')
    rename_dict = {
        "Country": "country_code",
        "Device Type": "device_type",
        "Login Successful": "is_login_success",
        "Is Attack IP": "is_attack_ip",
        "Is Account Takeover": "is_account_takeover",
    }

    df = df.rename(columns=rename_dict)

    return df

def drop_nan(df):
    print('[INFO] - Dropping NaN Values...')
    df = df.dropna()

    return df

def save_df(df):
    print('[INFO] - Saving Processed DataFrame...')
    processed_path = os.path.join(DATA_OUTPUT, 'processed/')
    os.makedirs(processed_path, exist_ok=True)
    df.to_csv(
        os.path.join(processed_path, 'processed.csv'),
        index=False,
        single_file=True
    )

    print('[SUCCESS] - DataFrame Saved Successfully!')

# Putting it all together
def process_features(df):
    df = extract_ip_octets(df)
    df = extract_hour_and_day_of_week(df)
    df = extract_browser_name(df)
    df = extract_os_name(df)
    df = drop_columns(df)
    df = rename_columns(df)
    df = drop_nan(df)

    return df

if __name__ == '__main__':
    ddf = load_data('rba-dataset.csv')
    ddf = process_features(ddf)
    save_df(ddf)