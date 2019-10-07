# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 22:43:48 2019

@author: LGULTRA
"""
#%%
import sys, os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np

#%%
root_dir = "C:\\Users\\LGULTRA\\Desktop\\kfood\\"

categories = ["Chicken", "Dolsotbab", "Kimchi",
              "Samgyeobsal", "SoybeanPasteStew"]

nb_classes = len(categories)
image_size = 64

#%%
len(categories)

#%% 데이터 불러오기
def load_dataset():
    x_train, x_test, y_train, y_test = np.load(root_dir + "koreanfood01.npy", allow_pickle = True)
    x_train = x_train.astype("float") / 256 #uint8
    x_test = x_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test
#%%
x_train, y_train, x_test, y_test = load_dataset()

#%%
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

type(y_train)
type(categories)
#%%
X_train_n = x_train.copy()
y_train_n = y_train.copy()

#%%
img_num = y_train[0:15, 0]
img_num
print('category = {}'.format(y_train[0:15, 0]))

#%%
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
#%%
type(y_train)


#%% => 왜 이거 안되지.....? 계속 TypeError: list indices must be integers or slices, not numpy.float32 뜬다..
print('Category = {}'.format(categories[y_train[0,0]]))

#%%

print('label = {}'.format([y_train[0:10,0]]))
#%%
# 10개의 이미지 데이터를 이용하여 그래프 확인
print('label = {}'.format([y_train[0:10,0]]))
fix, ax = plt.subplots(2, 5, figsize=(10,5))

for i in range(5):
  ax[0][i].imshow(np.reshape(x_train[i], (64, 64, 3)))
  ax[1][i].imshow(np.reshape(x_train[i+5], (64, 64, 3)))

#%%
print(y_train.shape), print(y_test.shape)

#%%
X = tf.placeholder(tf.float32, [None, 64, 64, 3])
Y = tf.placeholder(tf.float32, [None, 5])
keep_prob = tf.placeholder(tf.float32)

#%%
W1 = tf.Variable(tf.random_normal([3,3,3,64], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.dropout(L1, keep_prob)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L1)

#%%
W2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.dropout(L2, keep_prob)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2)

#%%
W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)

#%%
W4 = tf.Variable(tf.random_normal([3,3,128,32], stddev=0.01))
L4 = tf.nn.conv2d(L3, W4, strides=[1,1,1,1], padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.dropout(L4, keep_prob)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4)

#%%
W5 = tf.Variable(tf.random_normal([4*4*32, 256], stddev=0.01))
L5 = tf.reshape(L4, [-1, 4*4*32])
L5 = tf.matmul(L5, W5)
L5 = tf.nn.relu(L5)
print(L5)

#%%
W6 = tf.Variable(tf.random_normal([256,10],stddev=0.01))
model= tf.matmul(L5, W6)
model
#%%
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels =Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#%%
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#%%
batch_size = 20
total_batch = int(x_train.shape[0]/batch_size)
total_batch
epochs = 20
#%%
print(total_batch)
#%%
# 다음 배치를 읽어오기 위한 next_batch 유틸리티 함수를 정의합니다.

def next_batch(start, num, data, labels):
  """
  num 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
  # idx = np.arange(0 , len(data))
  # np.random.shuffle(idx)
  # idx = idx[:num]
  # data_shuffle = [data[i] for i in idx]
  # labels_shuffle = [labels[i] for i in idx]
  num 개수 만큼의 샘플과 레이블들을 리턴합니다.
  """
  
  data_X = data[start:start+num]
  data_y = categories[start:start+num]
  
  return np.asarray(data_X), np.asarray(data_y)

#%%
print(x_train.shape, y_train.shape)
batch_x, batch_y = next_batch(0,10,x_train, y_train)
print(batch_x.shape, batch_y.shape)
#%%
print('전체 입력 데이터 {}'.format(x_train.shape))
print('전체 출력 데이터 {}'.format(y_train.shape))

#%% ERROR: could not convert string to float: 'Chicken'
total_cost = 0
epoch = 0

for epoch in range(10+1):
  total_cost = 0
  for i in range(total_batch):
    batch_xs, batch_ys = next_batch(batch_size*i, batch_size, x_train, y_train)
    # 이미지 데이터를 CNN모델을 위한 자료형태인 [64,64,3]의 형태로 재구성합니다.//필요 없을 것 같지만..일단..
    batch_xs = batch_xs.reshape(-1, 64, 64, 3)
    _, cost_val= sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob:0.7})
    total_cost += cost_val
    
    if ( i == 0 or i == total_batch -1):
      print('data_step = {}, Avg cost = {:.2f}'.format(i, cost_val))
  print('epoch: {} total. cost = {:.2f}'.format(epoch, total_cost)) 
  
#%%