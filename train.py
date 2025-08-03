# Imports
import dask.dataframe as dd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import os

from utils.preprocessor import Preprocessor

# Config
DATA_PATH = './data/'
DATA_OUTPUT = './data/train/'
MODEL_DIR = 'src/models/train/'
SCALER_PATH = os.path.join(MODEL_DIR, 'min_max_scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
AUTOENCODER_PATH = os.path.join(MODEL_DIR, 'autoencoder_model.h5')

# Train Model
def split_test_train(df):
    normal_data = df[df['is_account_takeover'] == 0]
    anomalous_data = df[df['is_account_takeover'] == 1]

    train_data = normal_data.sample(frac=0.8, random_state=42)
    remaining_normal = dd.concat([normal_data, train_data]).drop_duplicates()
    test_data = dd.concat([remaining_normal, anomalous_data])

    train_data = train_data.compute()
    test_data = test_data.compute()

    X_train = train_data.drop(columns=['is_account_takeover'])
    X_test = test_data.drop(columns=['is_account_takeover'])
    y_test = test_data['is_account_takeover'].values

    return X_train, X_test, y_test

def create_autoencoder(X_train):
    # Get input shape
    input_dim = X_train.shape[1]

    # Define input
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)

    # Decoder
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    # Build the Model
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Model Summary
    autoencoder.summary()

    return autoencoder

def create_callbacks():
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)

    return early_stop, lr_scheduler

def train(model, X_train, epochs, batch_size, val_split, shuffle, callbacks):
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        shuffle=shuffle,
        callbacks=callbacks
    )

    return history

if __name__ == '__main__':

    processor = Preprocessor(
        'data/rba-dataset.csv',
        'data/train/preprocessed',
        'data/train/models',
        'logs/a.log'
    )
    processor.process_features()
    processor.encode_train()
    processor.scale_train()
    processor.cast_types(force_dask=True)
    processor.save_dataset()
    processor.save_scaler()
    processor.save_encoders()

    ddf = processor.get_df()

    # X_train, X_test, y_test = split_test_train(ddf)
    #
    # X_train = processor.handle_duplicates(X_train)
    #
    # autoencoder = create_autoencoder(X_train)
    # early_stop, lr_scheduler = create_callbacks()
    # history = train(
    #     autoencoder,
    #     X_train[:100_000],
    #     1,
    #     8,
    #     0.1,
    #     True,
    #     [early_stop, lr_scheduler]
    # )
    #
    # X_test_pred = autoencoder.predict(X_test, batch_size=256, verbose=1)
    # mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
    #
    # threshold = np.percentile(mse[y_test == 0], 95)
    # y_pred = (mse > threshold).astype(int)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))