import cv2, pafy
import os
import numpy as np 
import pandas as pd

from runDownload import downloadVideo
from runDownload import format10fps
from runDownload import download10

path = 'D:/PythonProjects/proj_faces/'

def runData():
    
    data = pd.read_csv(path + 'data.csv', delimiter=';')
    res = data.values
    
    for l in res:

        ## Verifica se ja foi realizado o download
        if l[5] == 0:
            downloadVideo(l[1], 'SpiderManNoWayHome')
            # format10fps(l[1],'SpiderManNoWayHome')
            # download10(l[1], l[0] + '/' + l[1], 'processed', l[3], l[4])
            # break
        

if __name__ == '__main__':
    
    print('aqui')
    # runData()