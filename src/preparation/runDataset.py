import cv2
import pafy
import os
import numpy as np 
import pandas as pd

from runDownload import downloadVideo
from runDownload import format10fps
from runDownload import download10

from runFaceDetectpy import choiceFolder 

# path = 'D:/PythonProjects/proj_faces/'

def runData():
    
    data = pd.read_csv('./data.csv', delimiter=';')
    res = data.values
    
    for l in res:
        downloadVideo(l[1], 'SpiderManNoWayHome')
        format10fps(l[1],'SpiderManNoWayHome')
        download10(l[1], l[0] + '/' + l[1], 'processed', l[3], l[4])


def detectFace():

    data = pd.read_csv('./data.csv', delimiter=';')
    res = data.values
    
    for l in res:

        choiceFolder(l[1], 'SpiderManNoWayHome')

        break

if __name__ == '__main__':
    
    # runData()

    detectFace()