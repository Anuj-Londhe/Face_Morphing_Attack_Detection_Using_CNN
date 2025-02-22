
import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as ms
import cv2
import sqlite3
import os
import numpy as np
import time
from tkvideo import tkvideo


global fn
fn = ""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="brown")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Morphing Detection")
video_label =tk.Label(root)
video_label.pack()
#read video to display on label
player = tkvideo("A.mp4", video_label,loop = 1, size = (w, h))
player.play()


label_l1 = tk.Label(root, text="Morphing Detection",font=("Times New Roman", 30, 'bold underline'),
                    background="#152238", fg="white", width=70, height=1)
label_l1.place(x=0, y=0)



################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def reg():
    from subprocess import call
    call(["python","registration.py"])

def log():
    from subprocess import call
    call(["python","Login.py"])
    
def window():
  root.destroy()


button1 = tk.Button(root, text="Login", command=log,width=14, height=1,font=('times', 20, ' bold '), bg="white", fg="black")
button1.place(x=100, y=160)

button2 = tk.Button(root, text="Registeration",command=reg,width=14, height=1,font=('times', 20, ' bold '), bg="white", fg="black")
button2.place(x=100, y=240)

button3 = tk.Button(root, text="Exit",command=window,width=11, height=1,font=('times', 20, ' bold '), bg="#152238", fg="white")
button3.place(x=120, y=320)

root.mainloop()