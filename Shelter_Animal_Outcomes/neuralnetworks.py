# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:19:59 2017

@author: 凯风
"""

import tensorflow as tf
import pandas as pd 
from sklearn.preprocessing import LabelBinarizer

# 要预测的
need_pre = pd.read_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/Processed_data/test.csv')
# 用来训练的
train = pd.read_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/Processed_data/train.csv')
target = pd.read_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/Processed_data/train_target.csv',header=None)

# 在读取数据的时候应该有一些参数设置，可以快速处理的，但是暂时还没掌握的辣么好~
need_pre.drop('Unnamed: 0',axis=1,inplace=True)
train.drop('Unnamed: 0',axis=1,inplace=True)
target.drop(0,axis=1,inplace=True)
target = LabelBinarizer().fit_transform(target)

# 添加层的函数
def addLayer(inputs,in_size,out_size,activation_function=None):
    # 初始化加权和偏移
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    
    # 计算输出
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    # 判断是否需要激励函数
    if activation_function is None:
        outputs = Wx_plus_b    # 线性
    else:
        outputs = activation_function(Wx_plus_b) # 传入的激励函数作用于输出
        
    return outputs

# placehold训练模型需要的数据
xs = tf.placeholder(tf.float32,[None,23])
ys = tf.placeholder(tf.float32,[None,5])

# 创建结构
prediction = addLayer(xs,23,5,activation_function=tf.nn.softmax)

# 优化目标
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))

# 训练的步骤
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 创建会话
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 跑起来~
for i in range(3000):
    sess.run(train_step,feed_dict={xs:train, ys:target})
    if i % 50 == 0:
        print (i)

predicted = sess.run(prediction, feed_dict={xs:need_pre})

output = pd.DataFrame(predicted,columns=['Return_to_owner','Transfer','Euthanasia','Died','Adoption'])
output.columns.names = ['ID']
output.index.names = ['ID']
output.index += 1
output.to_csv('D:/mygit/Kaggle/Shelter_Animal_Outcomes/model_pre/prediction_neuralnetwork.csv')
# 自己在文件中补充上ID吧。。。。