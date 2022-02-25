
import cv2, pafy
import numpy as np 
import os 
import math 
import csv
import requests
from urllib.request import urlopen
# import face_recognition
import sys
from pytube import YouTube
import os

from numpy import asarray
from numpy import savez_compressed

from moviepy.editor import *
import datetime

# ========================================================
## Local de armazenamento das imagens geradas            #
database_img = 'F:/FaceSpont/'                       #
# ========================================================

database = 'data/'
default_url = 'https://www.youtube.com/watch?v='

def downloadVideo(id, movie):
    
    try:
        if not os.path.exists(database_img + 'videos/original/' + movie):
            os.makedirs(database_img + 'videos/original/' + movie)
    except OSError:
        print('Error: Creating directory of data')
        
    try:
        
        url = "https://www.youtube.com/watch?v=" + id
        out = database_img + 'videos/original/' + movie
        
        urlopen(url)
        
        yt = YouTube(url)
        
        video = yt.streams.get_highest_resolution()    
        video.download(out)
        
        os.rename(out + '/' + yt.streams.get_highest_resolution().default_filename, out + '/' + id + '.mp4')
        
        print('Video ' + id + ' baixado!')
        return 'Video ' + id + ' baixado!'
        
    except:
        pass
            
    
    
def format10fps(id, movie):
    
    try:
        if not os.path.exists(database_img + 'videos/render/' + movie):
            os.makedirs(database_img + 'videos/render/' + movie)
    except OSError:
        print('Error: Creating directory of data')
    
    de      = database_img + 'videos/original/' + movie + '/' + id + '.mp4'
    para    = database_img + 'videos/render/' + movie + '/' + id + '.mp4'
    
    clip = VideoFileClip(de)
    clip.write_videofile(para, fps=10)
    
    return 'Video convertido!'

def download10(id, name_folder, tipo, inicio, fim):
    
    vidcap      = cv2.VideoCapture(database_img + 'videos/render/' + name_folder + '.mp4')
    
    clip        = VideoFileClip(database_img + 'videos/render/' + name_folder + '.mp4') 
    duration    = clip.duration    
    video_time = str(datetime.timedelta(seconds = int(duration)))
    # print("Duration : " + str(duration)) 
    # print(video_time)
    
    n_frames    = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = int(vidcap.get(cv2.CAP_PROP_FPS))
    
    caminho = tipo + '/reactions/' + name_folder + '/frames/'
    
    try:
        if not os.path.exists(database_img + caminho):
            os.makedirs(database_img + caminho)
    except OSError:
        print('Error: Creating directory of data')
        
    valini = inicio * 10
    valfim = fim * 10
     
    currentFrame = 0
    while(True):
        
        # Capture frame-by-frame
        ret, frame = vidcap.read()

        if not ret:
            break

        if currentFrame >= valini and currentFrame <= valfim:

            # Saves image of the current frame in jpg file
            name = database_img + caminho + '/frame_' + str(currentFrame) + '.jpg'
            # print ('Creating...' + name)
            cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    vidcap.release()
    cv2.destroyAllWindows()
    
    print('Frames de ' + id + ' baixados!')    
    return 'Frames de ' + id + ' baixados!'