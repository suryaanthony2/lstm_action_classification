import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, Input
from tensorflow.keras.optimizers import RMSprop
import json
import pathlib
from random import shuffle

parser = argparse.ArgumentParser(description="Train LSTM model using specified dataset")
parser.add_argument("-s", "--summary", action="store_true", help="print model summary")
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true", help="print verbose")
args = parser.parse_args()

if __name__ == "__main__":
    verb = 0
    if args.verbose:
        verb = 1
    
    with open(os.fspath(pathlib.Path(__file__).parent / "config.json"), "r") as f:
        cfg = json.load(f)

    x = np.load(os.fspath(pathlib.Path(__file__).parent / "dataset/train_x.npy"))
    y = np.load(os.fspath(pathlib.Path(__file__).parent / "dataset/train_y.npy"))
    x_val = np.load(os.fspath(pathlib.Path(__file__).parent / "dataset/val_x.npy"))
    y_val = np.load(os.fspath(pathlib.Path(__file__).parent / "dataset/val_y.npy"))

    input_x, input_y, input_z = np.split(x, 3)
    val_x, val_y, val_z = np.split(x_val, 3)

    ind_list = list(range(input_x.shape[0]))
    shuffle(ind_list)

    nn_input_x = Input(shape=input_x.shape[1:], name='x_input')
    nn_input_y = Input(shape=input_y.shape[1:], name='y_input')
    nn_input_z = Input(shape=input_z.shape[1:], name='z_input')

    mask_x = layers.Masking(mask_value=2.0)(nn_input_x)
    lstm_x = layers.LSTM(cfg["lstm"][0])(mask_x)

    mask_y = layers.Masking(mask_value=2.0)(nn_input_y)
    lstm_y = layers.LSTM(cfg["lstm"][1])(mask_y)

    mask_z = layers.Masking(mask_value=2.0)(nn_input_z)
    lstm_z = layers.LSTM(cfg["lstm"][2])(mask_z)

    concatenated = layers.concatenate([lstm_x, lstm_y, lstm_z])

    #dropout = layers.Dropout(0.5)(concatenated)

    dense = layers.Dense(cfg["dense"], activation='relu', )(concatenated)

    #dropout = layers.Dropout(0.5)(dense)

    out = layers.Dense(cfg["num_classes"], activation='softmax')(dense)

    model = Model([nn_input_x, nn_input_y, nn_input_z], out)

    model.compile(optimizer="rmsprop",
                loss="sparse_categorical_crossentropy",
                metrics=["acc"])

    if args.summary:
        model.summary()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=(cfg["patience"] * 0.6 ))

    early_stopping_monitor = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=cfg["patience"],
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )

    print("TRAINING MODEL, PLEASE WAIT!")

    model.fit(
        x=[input_x, input_y, input_z],
        y=y,
        verbose=verb,
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        validation_data=([val_x, val_y, val_z], y_val),
        callbacks=[early_stopping_monitor, reduce_lr]
    )

    model.save(os.fspath(pathlib.Path(__file__).parent.parent / "model/LSTM_model"), save_format="h5")

    print("TRAINING COMPLETE!")