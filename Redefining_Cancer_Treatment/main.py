# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:58:41 2017

@author: 凯风
"""

def loadData():
    
    print ('loading original data...')
    
    for name in ['training','test']:
        data1 = pd.read_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/%s_variants' % name)
        data2 = pd.read_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/%s_text' % name , sep='\|\|', engine='python', header=None, skiprows=1, names=["ID","Text"])
        if name == 'training':
            train = pd.merge(data1,data2,how='right',on='ID')
        if name == 'test':
            test = pd.merge(data1,data2,how='right',on='ID')
            
    return train,test

def preprocessOne(data):
 
    stopwords_english = nltk.corpus.stopwords.words('english')
    stemmer = nltk.stem.snowball.PortugueseStemmer(ignore_stopwords=True)
    
    with tqdm(total=len(data),desc='preprocessing',unit='cols') as pbar:
    
        for i,text in enumerate(data):
            # 分词
            words = nltk.tokenize.word_tokenize(text,language='english')
            # 去掉标点符号和长度小于2的单词
            words = [x for x in words if len(x) > 2]
            # 小写化
            words = [x.lower() for x in words]
 
            # 词干处理
            for index in range(len(words)):
                words[index] = stemmer.stem(words[index])

            # 删掉停用词
            words = (word for word in words if word not in stopwords_english)
                       
            newText = ''
            for word in words:
                newText = newText + ' ' + word
        
            data[i] = newText
            
            pbar.update(1)
            
    return data

def preprocessTwo(data,ntrain):
    print ('Tfidf vector ...')
    tfidf = TfidfVectorizer(stop_words = 'english')
    
    newTrainData = tfidf.fit_transform(data[:ntrain])
    newTextData = tfidf.transform(data[ntrain:])
    
    return sp.vstack([newTrainData,newTextData],format='csr')

def preprocessThree(data,ntrain):
    print ('svd ...')
    
    featureNum = 300
    enoughVar = False
    
    while not enoughVar:
        svd = TruncatedSVD(featureNum)
        trainData = svd.fit_transform(data[:ntrain])
        print (np.sum(svd.explained_variance_ratio_),'   ',featureNum)
        
        if (np.sum(svd.explained_variance_ratio_)) >= 0.95:
            enoughVar = True
            testData = svd.transform(data[ntrain:])
        else:
            featureNum = featureNum + 25
            continue
        
    return np.append(trainData,testData,axis=0)

def preprocessFour(data):
    
    for c in data.columns:
        
        if c in ['Gene','Variation']:
            lbl = LabelEncoder()
            data[c+'_lbl_enc'] = lbl.fit_transform(list(data[c].values))
            data[c+'_len'] = data[c].map(lambda x: len(str(x)))
        
        if c == 'Text':
            data[c+'_len'] = data[c].map(lambda x: len(str(x)))
    
    return data

if __name__ == '__main__':
    import pandas as pd
    import scipy.sparse as sp
    import numpy as np
    import nltk
    from tqdm import tqdm
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import LabelEncoder

    train ,test = loadData()
    target = train['Class']
    print ('saving target data ...')
    target.to_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/target.csv',index=False)
    del target
    
    # 丢掉没用的数据
    train.drop(['ID','Class'] ,axis=1 ,inplace=True)
    test.drop(['ID'] ,axis=1 ,inplace=True)    
    
    # 获取训练集和测试集的数量
    ntrain = train.shape[0]
    ntest = test.shape[0]
    
    # 合并数据
    allData = train.append(test)
    del train,test
    
    # 处理text,长文本数据
    newText = preprocessOne(allData['Text'])
    print (newText.shape)
    print ('saving long text file ...')
    newText.to_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/newText.csv',index=False, sep='|')
    allData.drop(['Text'],axis=1,inplace=True)

    '''
    # 读取数据的方法，主要是解决文件限制的
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
        
    aa = pd.read_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/newText.csv', sep='|', engine='python' ,header=None)
    '''
    
    # tfidf特征创建
    tfidfData = preprocessTwo(newText,ntrain)
    print (tfidfData.shape)
    filename = 'D:/mygit/Kaggle/Redefining_Cancer_Treatment/tfidfData'
    print ('saving tfidf data ...')
    np.savez_compressed(filename,data=tfidfData.data,indices=tfidfData.indices,indptr=tfidfData.indptr,shape=tfidfData.shape)

    # tfidf特征降维
    svdData = preprocessThree(tfidfData,ntrain)
    print (svdData.shape)
    print ('saving svd data ...')
    svdData = pd.DataFrame(svdData)
    svdData.to_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/svdData.csv',index=False)
    del tfidfData,svdData

    # 标称型特征处理,Gene和Variation
    allData = preprocessFour(allData)
    allData.drop(['Gene','Variation'],axis=1,inplace=True)
    allData.to_csv('D:/mygit/Kaggle/Redefining_Cancer_Treatment/otherData.csv')
