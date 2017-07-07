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
    import sys  
    import csv  
    maxInt = sys.maxsize  
    decrement = True  
      
    while decrement:  
        decrement = False  
        try:  
            csv.field_size_limit(maxInt)  
        except OverflowError:  
            maxInt = int(maxInt/10)  
            decrement = True  
        
    Data = pd.read_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/newText.csv', sep='|', engine='python' ,header=None)
    returnData = [] 
    for articleIndex in Data.index:
        returnData.append(Data[0][articleIndex])
    
    return returnData

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

def fit_model(corpus,id2word,num_topics=8):
    # 训练模型
    hdp = HdpModel(corpus=corpus, id2word=id2word)
    hdp.print_topics(num_topics)
    return hdp

def prediction_data(model,id2word,prediction_data):
    pre_doc = prediction_data
    vec_bow = id2word.doc2bow(pre_doc.split())
    vec_lda = model[vec_bow]
    bestSimilarity = {'theme':-1,'similarity':0}
    for i in range(len(vec_lda)):
        if vec_lda[i][1] > bestSimilarity['similarity']:
            bestSimilarity['similarity'] = vec_lda[i][1]
            bestSimilarity['theme'] = vec_lda[i][0]
    print('最相似的主题是：',bestSimilarity['theme'])
    print('相似度为：',bestSimilarity['similarity'])
    print(vec_lda)
    
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 清洗数据
X = loadData()
# 获取训练模型前所需要的数据
corpus,id2word = model_data(X)
# 训练模型
model = fit_model(corpus,id2word,num_topics=8)
# 预测数据
test_data = X[3000]
prediction_data(model,id2word,test_data)