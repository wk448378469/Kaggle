# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 10:47:27 2017

@author: 凯风
"""

import xgboost as xgb
import tensorflow as tf


class XgbWrapper(object):
    
    def __init__(self,seed=0,params = None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds',250)
    
    def train(self,x_train,y_train):
        dtrain = xgb.DMatrix(x_train,label=y_train)
        self.gbdt = xgb.train(self.param,dtrain,self.nrounds)
    
    def predict(self,x):
        return self.gbdt.predict(xgb.DMatrix(x))


class SklearnWrapper(object):
    
    def __init__(self,clf,seed=0,params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        
    def train(self,x_train,y_train):
        self.clf.fit(x_train,y_train)
    
    def predict(self,x_test):
        self.clf.predict(x_test)


class TensorflowWrapper(object):
    
    def __init__(self,n_step,input_size=500,learn_rate,activation_function=None):
        self.n_step = n_step
        self.input_size = input_size
        self.xs = tf.placeholder(tf.float32,[None,self.input_size])
        self.ys = tf.placeholder(tf.float32,[None,1])
        self.activation_function = activation_function
        self.add_layer()
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.pred),reduction_indices=[1]))
        self.train_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
        
    def add_layer(self):
        # 初始化weights和biases 
        Ws = self._weight_variable([self.input_size,1])
        bs = self._bias_variable([1,])
    
        # 计算神经元输出~
        Wx_plus_b = tf.matmul(self.input_size, Ws) + bs

        # axes，想要标准化的维度
        fc_mean, fc_var = tf.nn.moments(Wx_plus_b,axes=[0])
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001
    
        # 求均值和方差
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)
            
        mean, var = mean_var_with_update()
            
        # 标准化计算
        Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
    
        # 激活函数
        if self.activation_function is None:
            self.pred = Wx_plus_b
        else:
            self.pred = self.activation_function(Wx_plus_b)
    
        return self.pred

    def _weight_variable(self,shape):
        # 初始化变量weights
        initializer = tf.random_normal_initializer(mean=0,stddev=1.,)
        return tf.get_variable(shape=shape,initializer=initializer,name=name)
    
    def _bias_variable(self,shape):
        # 初始化变量bias
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape,initializer=initializer,name=name)
    

    def train(self,x_train,y_train):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        feed_dict = {self.xs:x_train,self.ys:y_train}
        for i in range(self.n_step):
            self.sess.run(self.train_op,feed_dict=feed_dict)
            
    def predict(self,x):
        return self.sess.run(self.pred,feed_dict={self.xs:x})