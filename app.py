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
        self.multi_file_path = ""
        self.res = ""

        self.geometry("1000x450")
        self.title("Karate app")
        sv_ttk.set_theme("dark")

        main_frame = Frame(self)
        main_frame.pack()

        self.create_frame(main_frame)

    def create_frame(self, main_frame):
        single_class = LabelFrame(main_frame, text="One Move Classification")
        single_class.grid(row=0, column=0)
        multi_class = LabelFrame(main_frame, text="Real Time Classification")
        multi_class.grid(row=1, column=0)
        self.single_cls_app_widgets(single_class)
        self.multi_cls_app_widgets(multi_class)
        
        for frame in main_frame.winfo_children():
            frame.grid_configure(padx=10, pady=20)

    def single_cls_app_widgets(self, frame):
        self.single_cls_btn_lbl = Label(frame, text="please choose mp4 file")
        self.single_cls_btn_lbl.grid(row=0, column=0)
        self.single_cls_btn = Button(frame, text="Select File", width = 40, command=self.get_file_path)
        self.single_cls_btn.grid(row=1, column=0)
        self.file_path_lbl = Label(frame, text="")
        self.file_path_lbl.grid(row=2, column=0)

        #button untuk melakukan inferensi
        self.btn_inf = Button(frame, text="lakukan inferensi", width = 40, command=self.inference)
        self.btn_inf.grid(row=0, column=1)
        self.lbl_inf = Label(frame, text="")
        self.lbl_inf.grid(row=1, column=1)

        for widget in frame.winfo_children():
            widget.grid_configure(padx=10, pady=5)

        self.update_single_text()

    def multi_cls_app_widgets(self, frame):
        #widget untuk milih kamera dan real time inference
        self.src_lbl = Label(frame, text="Choose source")
        self.src_lbl.grid(row=0, column=1)
        self.src_combo = ttk.Combobox(frame, values=["camera", "mp4"])
        self.src_combo.grid(row=1, column=1)
        self.camera_lbl = Label(frame, text="Choose camera")
        self.camera_lbl.grid(row=2, column=0)
        graph = FilterGraph()
        self.devices = graph.get_input_devices()
        self.camera_combo = ttk.Combobox(frame, values=self.devices)
        self.camera_combo.grid(row=3, column=0)

        self.multi_cls_btn = Button(frame, text="Please choose mp4 file", width = 40, command=lambda: self.get_file_path(single=False))
        self.multi_cls_btn.grid(row=2, column=2)
        self.multi_path_lbl = Label(frame, text="")
        self.multi_path_lbl.grid(row=3, column=2)

        self.btn_inf = Button(frame, text="inferensi real time", width = 40, command=self.inf_rt)
        self.btn_inf.grid(row=4, column=1)

        for widget in frame.winfo_children():
            widget.grid_configure(padx=10, pady=5)

        self.update_multi_text()

    def get_file_path(self, single=True):
        if single:
            self.file_path = filedialog.askopenfilename(title="Select a video File", filetypes=(("mp4", "*.mp4"), ))
        else:
            self.multi_file_path = filedialog.askopenfilename(title="Select a video File", filetypes=(("mp4", "*.mp4"), ))

    def inference(self):
        self.withdraw()
        if self.file_path == "":
            messagebox.showinfo("Error", "Select file first")
            return
        x = pose_detection.get_coordinates(self.file_path)
        input_x, input_y, input_z = np.split(x, 3)

        self.res, _ = prediction.classify(input_x, input_y, input_z, self.model)
        self.deiconify()


    def update_single_text(self):
        if self.file_path != "":
            self.file_path_lbl.config(text="file path = " + self.file_path)
        if self.res != "":
            self.lbl_inf.config(text="hasil prediksi = " + self.res)
        self.after(100, self.update_single_text)
    
    def update_multi_text(self):
        if self.multi_file_path != "":
            self.multi_path_lbl.config(text="file path = " + self.multi_file_path)
        self.after(100, self.update_multi_text)

    def inf_rt(self):
        self.withdraw()
        src = self.src_combo.get()
        cam = self.camera_combo.get()
        if src == "camera":
            if cam == "":
                messagebox.showinfo("Error", "Select camera first")
                return
            index = self.devices.index(cam)
            src_video = index
            flip = True
        elif src == "mp4":
            if self.multi_file_path == "":
                messagebox.showinfo("Error", "Select file mp4 first")
                return
            src_video = self.multi_file_path
            flip=False
        else:
           messagebox.showinfo("Error", "Select source first") 
        pose_detection.detect_real_time(src_video, self.model, flip=flip)
        self.deiconify()

if __name__ == "__main__":
    window = app()
    window.mainloop()