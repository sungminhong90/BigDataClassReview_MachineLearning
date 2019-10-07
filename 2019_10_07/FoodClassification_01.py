# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% 라이브러리 가져오기
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from PIL import Image
import os, glob
import numpy as np

#%% 데이터 불러오기

root_dir = "C:\\Users\\LGULTRA\\Desktop\\kfood\\"

#%% 카테고리 지정

categories = ["Chicken", "Dolsotbab", "Kimchi",
              "Samgyeobsal", "SoybeanPasteStew"]

nb_classes = len(categories)

#%% 이미지 크기 지정
image_width = 64
image_height = 64

#%% 이미지 데이터 X, 레이블 Y
## 데이터 변수
X = [] #이미지 데이터
Y = [] #레이블 데이터

for idx, category in enumerate(categories):
    image_dir = root_dir + category
    files = glob.glob(image_dir + '/' + '*.jpg')
    print(image_dir + "/" + "*.jpg")
    
    for i, f in enumerate(files):
        #이미지 로딩
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((image_width, image_height))
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)
        
X = np.array(X)
Y = np.array(Y)
print(X.shape, Y.shape)

#%% 데이터 셋 나누기
X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)

#%% 데이터 파일 저장
np.save(root_dir + "koreanfood01.npy", xy)

#%%
!dir