import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tkinter import *
from tkinter import filedialog
import pathlib
from lib import pose_detection, prediction
import numpy as np
from tensorflow.keras.models import load_model

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from pygrabber.dshow_graph import FilterGraph

import sv_ttk

class app(tk.Tk):
    def __init__(self):
        super().__init__()

        self.model = load_model(os.fspath(pathlib.Path(__file__).parent / "model/LSTM_model"), compile=False)

        self.file_path = ""
        self.res = ""

        self.geometry("1000x720")
        self.title("Karate app")
        sv_ttk.set_theme("dark")

        self.app_widgets()

    def app_widgets(self):
        self.btn = Button(self, text="Select File", width = 40, command=self.get_file_path)
        self.btn.pack(pady=10)
        self.lbl = Label(self, text="")
        self.lbl.pack(pady=40)

        #button untuk melakukan inferensi
        self.btn_inf = Button(self, text="lakukan inferensi", width = 40, command=self.inference)
        self.btn_inf.pack(pady=10)
        self.lbl_inf = Label(self, text="")
        self.lbl_inf.pack(pady=40)

        #widget untuk milih kamera dan real time inference
        self.camera_lbl = Label(self, text="Choose camera")
        self.camera_lbl.pack(pady=10)
        graph = FilterGraph()
        self.devices = graph.get_input_devices()
        self.camera_combo = ttk.Combobox(self, values=self.devices)
        self.camera_combo.pack()
        self.btn_inf = Button(self, text="inferensi real time", width = 40, command=self.inf_rt)
        self.btn_inf.pack(pady=10)

        self.update_text()

    def get_file_path(self):
        self.file_path = filedialog.askopenfilename(title="Select a video File", filetypes=(("mp4", "*.mp4"), ))

    def inference(self):
        if self.file_path == "":
            messagebox.showinfo("Error", "Select file first")
            return
        x = pose_detection.get_coordinates_lstm(self.file_path)
        input_x, input_y, input_z = np.split(x, 3)

        self.res, _ = prediction.classify(input_x, input_y, input_z, self.model)


    def update_text(self):
        if self.file_path != "":
            self.lbl.config(text="file path = " + self.file_path)
        if self.res != "":
            self.lbl_inf.config(text="hasil prediksi = " + self.res)
        self.after(100, self.update_text)

    def inf_rt(self):
        cam = self.camera_combo.get()
        if cam == "":
            messagebox.showinfo("Error", "Select camera first")
            return
        index = self.devices.index(cam)
        pose_detection.detect_real_time(index, self.model)

if __name__ == "__main__":
    window = app()
    window.mainloop()