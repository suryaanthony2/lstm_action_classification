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

class App(tk.Tk):
    def __init__(self):
        super().__init__()
    
        container = tk.Frame(self)
        sv_ttk.set_theme("dark")

        self.geometry("800x600")
        self.title("Karate app")

        container.pack(expand=True)

        self.frames = {}

        for F in (StartPage, SingleClass, MultiClass):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")
        
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.title_lbl = Label(self, text="KARATE CLASSIFICATION", font=("Times new roman", 30, "bold"))
        self.title_lbl.pack(pady=20)

        self.single_cls_btn = Button(self, text="Single Class Classification", width = 40, command=lambda: self.controller.show_frame("SingleClass"))
        self.single_cls_btn.pack(pady=10)

        self.multi_cls_btn = Button(self, text="Real Time Classification", width = 40, command=lambda: self.controller.show_frame("MultiClass"))
        self.multi_cls_btn.pack(pady=10)

class SingleClass(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.model = load_model(os.fspath(pathlib.Path(__file__).parent / "model/LSTM_model"), compile=False)
        self.controller = controller
        self.file_path = ""
        self.res = ""

        self.title_lbl = Label(self, text="Single Class Classification", font=("Times new roman", 30, "bold"))
        self.title_lbl.pack(pady=20)

        self.single_cls_btn = Button(self, text="Select File Path", width = 40, command=self.get_file_path)
        self.single_cls_btn.pack(pady=10)

        self.file_path_lbl = Label(self, text="")
        self.file_path_lbl.pack(pady=10)

        self.btn_inf = Button(self, text="Prediction", width = 40, command=self.inference)
        self.btn_inf.pack(pady=10)
        self.lbl_inf = Label(self, text="")
        self.lbl_inf.pack(pady=10)

        self.btn_back = Button(self, text="Back", width = 40, command=lambda: self.controller.show_frame("StartPage"))
        self.btn_back.pack(pady=10)

        self.update_text()

    def get_file_path(self):
        self.file_path = filedialog.askopenfilename(title="Select a video File", filetypes=(("mp4", "*.mp4"), ))

    def update_text(self):
        if self.file_path != "":
            self.file_path_lbl.config(text="File path = " + self.file_path)
        if self.res != "":
            self.lbl_inf.config(text="Prediction result = " + self.res)
        self.after(100, self.update_text)

    def inference(self):
        if self.file_path == "":
            messagebox.showinfo("Error", "Select file first")
            return
        x = pose_detection.get_coordinates(self.file_path)
        input_x, input_y, input_z = np.split(x, 3)

        self.res, _ = prediction.classify(input_x, input_y, input_z, self.model)

class MultiClass(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.model = load_model(os.fspath(pathlib.Path(__file__).parent / "model/LSTM_model"), compile=False)
        self.controller = controller
        self.file_path = ""

        self.title_lbl = Label(self, text="Real Time Classification", font=("Times new roman", 30, "bold"))
        self.title_lbl.pack(pady=20)

        self.src_lbl = Label(self, text="Choose source")
        self.src_lbl.pack(pady=5)
        self.src_combo = ttk.Combobox(self, values=["camera", "mp4"])
        self.src_combo.pack(pady=10)
        self.camera_lbl = Label(self, text="Choose camera")
        self.camera_lbl.pack(pady=5)
        graph = FilterGraph()
        self.devices = graph.get_input_devices()
        self.camera_combo = ttk.Combobox(self, values=self.devices)
        self.camera_combo.pack(pady=10)

        self.multi_cls_btn = Button(self, text="Please choose mp4 file", width = 40, command=self.get_file_path)
        self.multi_cls_btn.pack(pady=10)
        self.file_path_lbl = Label(self, text="")
        self.file_path_lbl.pack(pady=10)

        self.btn_inf = Button(self, text="Real Time Prediction", width = 40, command=self.inf_rt)
        self.btn_inf.pack(pady=10)

        self.btn_back = Button(self, text="Back", width = 40, command=lambda: self.controller.show_frame("StartPage"))
        self.btn_back.pack(pady=10)

        self.update_text()

    def get_file_path(self):
        self.file_path = filedialog.askopenfilename(title="Select a video File", filetypes=(("mp4", "*.mp4"), ))

    def update_text(self):
        if self.file_path != "":
            self.file_path_lbl.config(text="File path = " + self.file_path)
        self.after(100, self.update_text)

    def inf_rt(self):
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
            if self.file_path == "":
                messagebox.showinfo("Error", "Select file mp4 first")
                return
            src_video = self.file_path
            flip=False
        else:
           messagebox.showinfo("Error", "Select source first") 
           return
        pose_detection.detect_real_time(src_video, self.model, flip=flip)

if __name__ == "__main__":
    window = App()
    window.mainloop()