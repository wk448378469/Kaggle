# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:21:05 2017

@author: kaifeng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

train = pd.read_csv('C:/Users/carne/Desktop/train.csv')
test_images = (pd.read_csv('C:/Users/carne/Desktop/test.csv').values).astype('float32')

train_images = (train.ix[:,1:].values).astype('float32')
train_lables = train.ix[:,0].values.astype('int32') 

# 把这些数字转化成图片看看
train_images = train_images.reshape(train_images.shape[0],28,28) # 每个图像的高度是28*28
for i in range(6,9):
    plt.subplot(330 + (i+1))
    plt.imshow(train_images[i],cmap=plt.get_cmap('gray'))
print (train_lables[6:9])
# 还是挺准确的.....

# 再把train_images 变回去吧

train_images = train_images.reshape(train_images.shape[0],(28*28))


# 数据的预处理，就是标准化一下
train_images = train_images/255
test_images = test_images/255

# one hot code
train_lables = to_categorical(train_lables)  # 但是深度学习为毛要在这里做呢？
num_classes = train_lables.shape[1]   # 分了10类

plt.title(train_lables[9])
plt.plot(train_lables[9])
plt.xticks(range(10))
# 就说明这个数字是三

# 设计神经网络的结构了开始
seed = 43
np.random.seed(seed) #种子生成器？
model = Sequential()
model.add(Dense(32,activation='relu',input_dim = (28*28)))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))


'''
    # 训练前的一些事情
    1、代价函数
    2、如何优化
    3、具体哪些指标
'''
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

# 开始训练
history = model.fit(train_images,train_lables,validation_split=0.05,epochs=25,batch_size=64)

# 得到模型的一些评价结果
history_dict = history.history
loss_values = history_dict['loss']      # 损失值，虽然我还不知道官方的叫法
val_loss_values = history_dict['val_loss']  # 另外一个损失值
epochs = range(1,len(loss_values) + 1)
# 画出来这些值
plt.plot(epochs,loss_values,'bo')
plt.plot(epochs,val_loss_values,'^')
plt.xlabel('Epochs')
plt.ylabel('Loss')          # 随着时间的推移，损失越来越小

plt.clf() # 清理掉作图空间

acc_values = history_dict['acc']        # 准确度
val_acc_values = history_dict['val_acc']    #另外一个准确度
# 画出来这些值
plt.plot(epochs,acc_values,'bo')
plt.plot(epochs,val_acc_values,'^')
plt.xlabel('Epochs')
plt.ylabel('Loss')              # 随着时间的推移，准确度越来越高

# 调整下参数
model = Sequential()
model.add(Dense(64,activation='relu',input_dim = (28*28)))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer=Adam(lr = 1e-3),loss='categorical_crossentropy',metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=2, mode='auto'),
            ModelCheckpoint('mnist.h5', monitor='val_loss', save_best_only=True, verbose=0)]



history = model.fit(train_images,train_lables,validation_split=0.125,epochs=200,batch_size=64,callbacks=callbacks)
# 好像结果更差了......


# 然后在测试集上输出结果就可以了
predictions = model.predict_classes(test_images,verbose=0)

# 输出结果
submission = pd.DataFrame({
        'ImageId':list(range(1,len(predictions)+1)),
        'Label':predictions
        })

submission.to_csv('C:/Users/carne/Desktop/submission.csv',index=False,header=True)


