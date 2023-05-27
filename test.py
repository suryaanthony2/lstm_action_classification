import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tkinter import *
from tkinter import filedialog
import pathlib
from lib import pose_detection, prediction
import numpy as np
from tensorflow.keras.models import load_model

root = Tk()
# Hide the window
root.withdraw()

path = filedialog.askopenfilename()

root.destroy()

x = pose_detection.get_coordinates_lstm(path)
input_x, input_y, input_z = np.split(x, 3)

model = load_model(os.fspath(pathlib.Path(__file__).parent / "model/LSTM"), compile=False)

res = prediction.classify(input_x, input_y, input_z, model)

window = Tk()
window.geometry("400x100")
window.title("Hasil")
Label(window, text="Hasil prediksi = " + res, font=('Aerial 16')).pack()
window.mainloop()