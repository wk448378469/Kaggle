# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:06:40 2017

@author: 凯风
"""
import pandas as pd

from gensim.models.hdpmodel import HdpModel
from gensim.models.tfidfmodel import TfidfModel
from gensim import matutils,corpora
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import logging


def loadData():
    print ('loading original data...')
    for name in ['training','test']:
        data1 = pd.read_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/%s_variants' % name)
        data2 = pd.read_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/%s_text' % name , sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
        if name == 'training':
            train = pd.merge(data1,data2,how='left',on='ID')
        if name == 'test':
            test = pd.merge(data1,data2,how='left',on='ID')
        del data1,data2

    return train,test

def clean_data(data):
    new_data = []
    for each in data:
        # 小写化
        each = each.lower()
        # 删除固定格式的无用的字符串
        del_str_position = each.find('lines:') + 6
        each = each[del_str_position:]
        new_data.append(each)
    return new_data

def model_data(data):
    # 文集向量化
    vec = CountVectorizer(min_df=5,stop_words='english')
    train_X = vec.fit_transform(data)
    corpus = matutils.Sparse2Corpus(train_X)
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # 词典
    dictionary = corpora.Dictionary()
    data_seg = []
    for text in data:
        x = nltk.tokenize.word_tokenize(text,language='english')
        data_seg.append(x)
    dictionary.add_documents(data_seg)
    return corpus_tfidf,dictionary

def fit_model(corpus,id2word,num_topics=20):
    # 训练模型
    hdp = HdpModel(corpus=corpus, id2word=id2word)
    hdp.print_topics(num_topics)
    return hdp

def prediction_data(model,id2word,prediction_data):
    pre_doc = prediction_data
    vec_bow = id2word.doc2bow(pre_doc.lower().split())
    vec_lda = model[vec_bow]
    bestSimilarity = {'theme':-1,'similarity':0}
    for i in range(len(vec_lda)):
        if vec_lda[i][1] > bestSimilarity['similarity']:
            bestSimilarity['similarity'] = vec_lda[i][1]
            bestSimilarity['theme'] = vec_lda[i][0]
    print('最相似的主题是：',bestSimilarity['theme'])
    print('相似度为：',bestSimilarity['similarity'])
    
# 清洗数据
X = clean_data(train['Text'])
# 获取训练模型前所需要的数据
corpus,id2word = model_data(train['Text'])
# 训练模型
model = fit_model(corpus,id2word,num_topics=20)
# 预测数据
test_data = test['Text'][0]
prediction_data(model,id2word,test_data)