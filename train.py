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

# Preprocessing
def encode(df):
    cols_to_encode = ['country_code', 'device_type', 'browser_name', 'os_name']
    encoders = {}

    for col in cols_to_encode:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col].astype(str))
        encoders[col] = label_encoder

    bool_cols = ['is_login_success', 'is_attack_ip', 'is_account_takeover']
    for col in bool_cols:
        df[col] = df[col].astype(int)

    return df, encoders

def scale(df):
    cols_to_scale = [
        'country_code', 'device_type', 'is_login_success', 'is_attack_ip',
        'login_hour', 'login_day', 'browser_name', 'os_name'
    ]

    scaler = MinMaxScaler()

    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df, scaler

def cast_type(df):
    df['country_code'] = df['country_code'].astype(np.float32)
    df['device_type'] = df['device_type'].astype(np.float32)
    df['is_login_success'] = df['is_login_success'].astype(np.float32)
    df['is_attack_ip'] = df['is_attack_ip'].astype(np.float32)
    df['login_hour'] = df['login_hour'].astype(np.float32)
    df['login_day'] = df['login_day'].astype(np.float32)
    df['browser_name'] = df['browser_name'].astype(np.float32)
    df['os_name'] = df['os_name'].astype(np.float32)
    df['ip_1'] = df['ip_1'].astype(np.float32)
    df['ip_2'] = df['ip_2'].astype(np.float32)
    df['ip_3'] = df['ip_3'].astype(np.float32)
    df['ip_4'] = df['ip_4'].astype(np.float32)

    return df

def split_test_train(df):
    normal_data = df[df['is_account_takeover'] == 0]
    anomalous_data = df[df['is_account_takeover'] == 1]

    train_data = normal_data.sample(frac=0.8, random_state=42)
    remaining_normal = dd.concat([normal_data, train_data]).drop_duplicates()
    test_data = dd.concat([remaining_normal, anomalous_data])

    X_train = train_data.drop(columns=['is_account_takeover'])
    y_train = test_data.drop(columns=['is_account_takeover'])
    y_test = test_data['is_account_takeover'].values

    return X_train, y_train, y_test


if __name__ == '__main__':
    ddf = load_data('rba-dataset.csv')
    ddf = process_features(ddf)
    ddf = ddf.compute()
    ddf, encoders = encode(ddf)
    ddf, scaler = scale(ddf)
    ddf = cast_type(ddf)
    X_train, y_train, y_test = split_test_train(ddf)

