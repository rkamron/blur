import os
import cv2
import glob
import pickle
import numpy as np 
import pandas as pd 
#import tensorflow as tf
#from tensorflow import keras
import matplotlib.pyplot as plt

file_path = "../sample-results/"
for filename in glob.iglob("../data/videos//*.mp4", recursive = True):
    vidcap = cv2.VideoCapture(filename)
    print(filename)
    count=0
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(1,length,60):
        vidcap.set(1,i)
        ret, still = vidcap.read()
        cv2.imwrite(os.path.join(file_path,filename+"frame"+str(i)+".png"),still)