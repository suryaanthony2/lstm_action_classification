import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Input
import matplotlib.pyplot as plt
import json
import pathlib
from random import shuffle

parser = argparse.ArgumentParser(description="Train LSTM model using specified dataset")
parser.add_argument("-s", "--summary", action="store_true", help="print model summary")
parser.add_argument("-v", "--verbose", action="store_true", help="print verbose")
parser.add_argument("-g", "--graph", action="store_true", help="display model training graph")
args = parser.parse_args()

if __name__ == "__main__":
    verb = 0
    val = True

    if args.verbose:
        verb = 1
    
    with open(os.fspath(pathlib.Path(__file__).parent / "config.json"), "r") as f:
        cfg = json.load(f)

    try:
        x = np.load(os.fspath(pathlib.Path(__file__).parent / "dataset/train_x.npy"))
        y = np.load(os.fspath(pathlib.Path(__file__).parent / "dataset/train_y.npy"))
        x_val = np.load(os.fspath(pathlib.Path(__file__).parent / "dataset/test_x.npy"))
        y_val = np.load(os.fspath(pathlib.Path(__file__).parent / "dataset/test_y.npy"))
        val_x, val_y, val_z = np.split(x_val, 3)
    except:
        x = np.load(os.fspath(pathlib.Path(__file__).parent / "dataset/train_x.npy"))
        y = np.load(os.fspath(pathlib.Path(__file__).parent / "dataset/train_y.npy"))
        print("Validation dataset not found, Early stopping not used")
        print("Using tarining dataset only")
        epochs = int(input("Enter the number of training epochs: "))
        val = False
    
    input_x, input_y, input_z = np.split(x, 3)

    ind_list = list(range(input_x.shape[0]))
    shuffle(ind_list)

    nn_input_x = Input(shape=input_x.shape[1:], name='x_input')
    nn_input_y = Input(shape=input_y.shape[1:], name='y_input')
    nn_input_z = Input(shape=input_z.shape[1:], name='z_input')

    mask_x = layers.Masking(mask_value=2.0, name="x_mask")(nn_input_x)
    lstm_x = layers.LSTM(cfg["lstm"][0], name="x_lstm")(mask_x)

    mask_y = layers.Masking(mask_value=2.0, name="y_mask")(nn_input_y)
    lstm_y = layers.LSTM(cfg["lstm"][1], name="y_lstm")(mask_y)

    mask_z = layers.Masking(mask_value=2.0, name="z_mask")(nn_input_z)
    lstm_z = layers.LSTM(cfg["lstm"][2], name="z_lstm")(mask_z)

    concatenated = layers.concatenate([lstm_x, lstm_y, lstm_z], name="concatenate")

    dropout = layers.Dropout(0.5, name="dropout")(concatenated)

    dense = layers.Dense(cfg["dense"], activation='relu', name="dense")(dropout)

    out = layers.Dense(cfg["output"], activation='softmax', name="output")(dense)

    model = Model([nn_input_x, nn_input_y, nn_input_z], out)

    model.compile(optimizer=cfg["optimizer"],
                loss="sparse_categorical_crossentropy",
                metrics=["acc"])

    if args.summary:
        model.summary()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=30)

    early_stopping_monitor = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=50,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )

    print("TRAINING MODEL, PLEASE WAIT!")

    if val:
        history = model.fit(
            x=[input_x, input_y, input_z],
            y=y,
            verbose=verb,
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            validation_data=([val_x, val_y, val_z], y_val),
            callbacks=[early_stopping_monitor, reduce_lr]
        )
    else:
        history = model.fit(
            x=[input_x, input_y, input_z],
            y=y,
            verbose=verb,
            batch_size=cfg["batch_size"],
            epochs=epochs
        )     

    print("TRAINING COMPLETE!")

    model.save(os.fspath(pathlib.Path(__file__).parent.parent / "model/LSTM_model"), save_format="h5")

    if args.graph:
        figure, axis = plt.subplots(1, 2, figsize=(10, 6))
        axis[0].plot(history.history['acc'],'--')
        axis[0].plot(history.history['val_acc'])
        axis[0].set_title('model accuracy')
        axis[0].set(xlabel='epoch', ylabel='accuracy')
        axis[0].legend(['train', 'test'], loc='upper left')

        axis[1].plot(history.history['loss'],'--')
        axis[1].plot(history.history['val_loss'])
        axis[1].set_title('model loss')
        axis[1].set(xlabel='epoch', ylabel='loss')
        axis[1].legend(['train', 'test'], loc='upper left')
        
        plt.show()