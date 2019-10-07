# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 22:24:54 2019

@author: LGULTRA
"""
#%%
# 모델을 불러와서 이제 테스트 해보기!!!!
#%%
import sys, os
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.models import load_model
from PIL import Image
import numpy as np

#%%
root_dir = "C:\\Users\\LGULTRA\\Desktop\\kfood\\"

#%%
test_dir = root_dir + 'test_img/'
file_list = os.listdir(test_dir)

#%%
image_files = []

for i in file_list:
    sample = root_dir + "test_img/" + i
    image_files.append(sample)
    
#%%
image_size = 64
nb_classes = len(image_files)
categories = ["Chicken", "Dolsotbab", "Kimchi",
              "Samgyeobsal", "SoybeanPasteStew"]

#%%
X = []
files = []
#이미지 불러오기
for fname in image_files:
    print(fname)
    img = Image.open(fname)
    img = img.convert('RGB')
    img = img.resize((image_size, image_size))
    in_data = np.asarray(img)
    in_data = in_data.astype('float') / 256
    X.append(in_data)
    files.append(fname)
    
print(X)
print(files)

#%%
X = np.array(X)

#모델 파일 읽어오기
model = load_model(root_dir + "koreanfood01_model.h5")

#예측실행
pre = model.predict(X)

#예측 결과 출력
for i, p in enumerate(pre):
    y = p.argmax()
    print('입력:', files[i])
    print('예측:', "[", y, "]", categories[y], "/Score", p[y])

    
#%%