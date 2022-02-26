
import cv2 
import pafy
import numpy as np 
from PIL import Image
import face_recognition
import os 
import natsort 
from os import path
from mtcnn.mtcnn import MTCNN
import pandas as pd

import dlib
import argparse
import time

# from runPaths import caminhoArquivos
# from runPaths import readMovies

path_dataset = './dataset/processed/reactions/'

ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True, help='path to image file')
ap.add_argument('-w', '--weights', default='model/mmod_human_face_detector.dat',
                help='path to weights file')
args = ap.parse_args()

## Opencv
def detectFace1(img, folder, coord, comp):

    try:
        if not os.path.exists(path_dataset + folder + '/faces/'):
            os.makedirs(path_dataset + folder + '/faces/')
    except OSError:
        print('Error: Creating directory of data')

    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml') # pylint: disable=no-member

    image = cv2.imread(path_dataset + folder + '/frames/' + img + '.jpg') # pylint: disable=no-member

    ## Realiza a anulação do video frame de cada imagem, com o objetivo de não buscar as faces do video trailer
    cv2.rectangle(image, coord[0], coord[1], (0,0,0), -1)  # pylint: disable=no-member

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member

    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
    
    face_crop = []
    for (x, y, w, h) in faces: 

        cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2) # pylint: disable=no-member
        face_crop.append(gray[y:y+h, x:x+w])

    count = 0
    for face in face_crop:

        count = count + 1
        img_face = path_dataset + folder + '/faces/face1/' + img + '_' + str(count) + '.jpg'
        
        cv2.imwrite(img_face, face) # pylint: disable=no-member

        # cv2.imshow('face',face) # pylint: disable=no-member
        # cv2.waitKey(0) # pylint: disable=no-member
        # cv2.imwrite(img_face, face) # pylint: disable=no-member

    # cv2.imshow('img', image) # pylint: disable=no-member
    # cv2.waitKey() # pylint: disable=no-member

## Face Recognition
def detectFace2(img, folder, coord):

    try:
        if not os.path.exists(path_dataset + folder + '/faces/'):
            os.makedirs(path_dataset + folder + '/faces/')
    except OSError:
        print('Error: Creating directory of data')

    path_img = path_dataset + folder + '/frames/'
    # path_face = path_dataset + folder + '/faces/'

    imagem = face_recognition.load_image_file(path_img + img + '.jpg')

    cv2.rectangle(imagem, coord[0], coord[1], (0,0,0), -1)  # pylint: disable=no-member

    face_locations = face_recognition.face_locations(imagem)

    print("Encontradas {} faces no video".format(len(face_locations)))

    ## Roda a abertura de todas as faces do frame em questão
    count = 0
    for fl in face_locations:

        count = count + 1
        top, right, bottom, left = fl 
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        face_image = imagem[top:bottom, left:right]

        img_path = path_dataset + folder + '/faces/' + img + '_' + str(count) + '.jpg'
        if (len(face_locations) > 0):
            print('Creating... ' + img_path)
            cv2.imwrite(img_path, face_image) # pylint: disable=no-member

        cv2.imshow('img', face_image) # pylint: disable=no-member
        cv2.waitKey() # pylint: disable=no-member

## MTCNN
def detectFace3(img, folder, coord):

    imagem = cv2.imread(path_dataset + folder + '/frames/' + img + '.jpg') # pylint: disable=no-member

    cv2.rectangle(imagem, coord[0], coord[1], (0,0,0), -1)  # pylint: disable=no-member

    detector = MTCNN()

    result = detector.detect_faces(imagem)

    # print(detector.detect_faces(imagem))

    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    cv2.rectangle(imagem, # pylint: disable=no-member
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                (0,155,255),
                2) 

    cv2.circle(imagem, (keypoints['left_eye']), 2, (0,155,255), 2) # pylint: disable=no-member
    cv2.circle(imagem, (keypoints['right_eye']), 2, (0,155,255), 2) # pylint: disable=no-member
    cv2.circle(imagem, (keypoints['nose']), 2, (0,155,255), 2) # pylint: disable=no-member
    cv2.circle(imagem, (keypoints['mouth_left']), 2, (0,155,255), 2) # pylint: disable=no-member
    cv2.circle(imagem, (keypoints['mouth_right']), 2, (0,155,255), 2) # pylint: disable=no-member

    cv2.imshow('img', imagem) # pylint: disable=no-member
    cv2.waitKey() # pylint: disable=no-member

## Hog
def detectFace4(img, folder, coord, comp):

    try:
        if not os.path.exists(path_dataset + folder + '/faces/'):
            os.makedirs(path_dataset + folder + '/faces/')
    except OSError:
        print('Error: Creating directory of data')

    imagem = cv2.imread(path_dataset + folder + '/frames/' + img + '.jpg') # pylint: disable=no-member

    ## Realiza a anulação do video frame de cada imagem, com o objetivo de não buscar as faces do video trailer
    cv2.rectangle(imagem, coord[0], coord[1], (0,0,0), -1)  # pylint: disable=no-member

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member

    # initialize hog + svm based face detector
    hog_face_detector = dlib.get_frontal_face_detector()

    faces_hog = hog_face_detector(imagem, 1)

    ## Cria pasta de acordo com o numero de faces encontradas
    x = range(0, len(faces_hog))
    for n in x:
        try:
            if not os.path.exists(path_dataset + folder + '/faces/face' + str(n + 1)):
                os.makedirs(path_dataset + folder + '/faces/face' + str(n + 1))
        except OSError:
            print('Error: Creating directory of data')

    
    for f in faces_hog:

        arTemp = []
        face_crop = []
            
        x = f.left()
        y = f.top()
        w = f.right() - x
        h = f.bottom() - y
            

        for c in comp:

            dist = np.linalg.norm(c - x)
            arTemp.append(dist)

            face_crop.append(gray[y:y+h, x:x+w])  
  
        if len(arTemp) > 0:
            menor = arTemp.index(min(arTemp))

            img_face = path_dataset + folder + '/faces/face' + str(menor+1) + '/' + img + '_' + str(menor + 1) + '.jpg'
            tfaces = np.array(face_crop[menor]) 

            if tfaces.size > 0:
                cv2.imwrite(img_face, face_crop[menor]) # pylint: disable=no-member

## CNN Face Detection
def detectFace5(img, folder, coord, comp):

    print(path_dataset + folder + '/frames/' + img + '.jpg')
    imagem = cv2.imread(path_dataset + folder + '/frames/' + img + '.jpg') # pylint: disable=no-member

    cv2.rectangle(imagem, coord[0], coord[1], (0,0,0), -1)  # pylint: disable=no-member

    # initialize cnn based face detector with the weights
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)

    faces_cnn = cnn_face_detector(imagem, 1)

    # loop over detected faces
    for face in faces_cnn:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        # draw box over face
        cv2.rectangle(imagem, (x,y), (x+w,y+h), (0,255,0), 2) # pylint: disable=no-member

    cv2.imshow('img', imagem) # pylint: disable=no-member
    cv2.waitKey() # pylint: disable=no-member
    

## Função que ignora o quadro do video que esta sendo reagido
# Tira-se um PRINT do exato quadro que se inicia o video durante a reação. 
# Esse print deve ser tirado do frame inicial da reação (e nomeado como frame_tm.jpg), o mesmo que foi anotado no arquivo de log_search.txt
# Parametros (frame inicial, print retirado, pasta do video, filme)
# Retorno - Coordenadas do frame, que serão IGNORADAS durante o processo de detecção de faces
def ignoreVideoFrame(img, template, pasta, filme):
    
    path_base = 'F:/Dataset_Faces/processed/reactions/' +  filme + '/' + pasta + '/frames/' + img
    path_compare = 'F:/Dataset_Faces/processed/reactions/' +  filme + '/' + pasta + '/frames/' + template + '.jpg'
    
    img = cv2.imread(path_base,0) # pylint: disable=no-member
    img2 = img.copy()
    template = cv2.imread(path_compare,0) # pylint: disable=no-member
    w, h = template.shape[::-1]

    imgt = img2.copy()
    method = eval('cv2.TM_CCOEFF')

    res = cv2.matchTemplate(imgt,template,method) # pylint: disable=no-member
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) # pylint: disable=no-member

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]: # pylint: disable=no-member
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    ## Retorna as coordenadas do video frame, para realizar a exclusao de possiveis faces
    return top_left, bottom_right

## Parametros (pasta do filme, inicio da reação, fim da reação, filme)
def choiceFolder(pasta, filme):

    entries = os.listdir(path_dataset + '/' + filme + '/' + pasta + '/frames/')
    nentries = natsort.natsorted(entries,reverse=False)

    print(nentries)
    exit()
    
    r = ignoreVideoFrame(nentries[0], 'frame_tm', pasta, filme)
    # r1 = ignoreVideoFrame(nentries[0], 'frame_tm1', pasta, filme)
            
    c = 1
    temp = []
    for ent in nentries:

        ent = ent.split('.')
        l = ent[0].split('_')

        if 'tm' not in l[1]:

            ## Função de detecção de Faces 
            ## É possivel utilizar 5 funções diferentes
            # detectFace1 - OpenCV
            # detectFace2 - Face recognition lib
            # detectFace3 - MTCNN
            # detectFace4 - Hog
            # detectFace5 - CNN Face detection
            
            ## Coleta cooredenadas das faces iniciais
            if c == 1:
                comp = getFirstCoord(ent[0], pasta, filme, r)                
                temp = comp

            ## Parametros (imagem que deseja detectar a face, pasta do frame, coordenadas que se deseja ignorar)
            detectFace4(ent[0], filme + '/' + pasta, r, temp)
            # break
            ## Uma nova pasta sera criada (/faces), com todas as faces geradas para analise

            # if c == 10:
            #     break        
            c += 1


def getFirstCoord(img, pasta, filme, coord):

    imagem = cv2.imread(path_dataset + filme + '/' + pasta + '/frames/' + str(img) + '.jpg') # pylint: disable=no-member

    ## Realiza a anulação do video frame de cada imagem, com o objetivo de não buscar as faces do video trailer
    cv2.rectangle(imagem, coord[0], coord[1], (0,0,0), -1)  # pylint: disable=no-member
    
    # cv2.imshow('img', imagem) # pylint: disable=no-member
    # cv2.waitKey() 

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # pylint: disable=no-member

    # initialize hog + svm based face detector
    hog_face_detector = dlib.get_frontal_face_detector()

    faces_hog = hog_face_detector(imagem, 1)

    arFace = []
    for face in faces_hog:

        x = face.left()
        arFace.append(x)
    
    return arFace

if __name__ == '__main__':

    # Percorre pasta com frames dos videos ja baixados
    # entries = os.listdir(path_dataset + 'Loki')

    # for ent in entries:

    #     if 'vWEBkpXZa3U' in ent or 'zxfJYDN6Boc' in ent:
    #         print('Detectando faces em ' + ent  + '...')
    #         choiceFolder(ent, 'Loki')
    #         print('Faces de ' + ent + ' criadas com sucesso!')
    
    
    caminho = 'D:/PythonProjects/proj_faces/'
    
    data = pd.read_csv(caminho + '/data/data.csv', delimiter=';')
    res = data.values
    
    for l in res:
                        
        ## Verifica se ja foi realizado o download
        if l[5] == 0:
            if 'Loki' in l[0]:
                if 'U5Ac3F26IUw' in l[1]:
                    print('Detectando faces em ' + l[1]  + '...')
                    choiceFolder(l[1], 'Loki')
                    print('Faces de ' + l[1] + ' criadas com sucesso!')

