# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:25:53 2017

@author: kaifeng
"""

# 导入库
import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron  #感知器
from sklearn.linear_model import  SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split    #分割数据集的
from sklearn.metrics import precision_recall_curve,classification_report,roc_auc_score # 其他评估指标
from sklearn.grid_search import GridSearchCV    #交叉验证的 


#导入数据
train_df = pd.read_csv('C:/Users/carne/Desktop/train.csv')
test_df = pd.read_csv('C:/Users/carne/Desktop/test.csv')

#查看训练集的features
print (train_df.columns.values)
train_df.head() #前五行
train_df.tail() #后五行

#查看数据类型
train_df.info()
test_df.info()

#查看统计数据
train_df.describe()
train_df.describe(include = ['O'])
'''
1、可能会删除掉Ticket 这个特征，因为票号对预测没有什么作用，而且票号的样本数量应该和总数一样，但是重复性居然有210个
2、Cabin因为有太多的缺失值也可能去掉
3、PassengerId也可能没什么用丢掉
4、名字对是否生存来说也没什么卵用
'''

'''
根据kaggle的说明可能有一下几个feature对生存影响较大
    1、性别=female的生存下去的几率更高？
    2、年龄小于<N的生存下去的几率更高？
    3、Pclass低的，买的票好，所以几率更高？
    验证一下猜想
'''

train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# 1的概率0.629630     2的概率0.472826      3的概率0.242363   

train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived')
# male生存的概率0.188908     female的概率0.742038

train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# 是否有兄弟姐妹好像没有什么帮助  字段里面还包含配偶，所以也不好说

train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#　是否有父母小孩的好像也没有什么帮助



#可视化的方法查看数据的相关信息
g = sns.FacetGrid(train_df,col = 'Survived')
g.map(plt.hist,'Age',bins = 20)
'''
观察图像好像可能看出：
    1、年级比较小于4岁的存活的概率挺高的（一根柱子是4岁的区间）
    2、80左右的都活了下来
    3、大多数的年级在15-35左右
'''

grid = sns.FacetGrid(train_df, col ='Survived', row ='Pclass' ,size=2.2, aspect=1.6)
grid.map(plt.hist,'Age',alpha = 0.5,bins = 20)
grid.add_legend()
'''
观察图像好像可能看出：
    1、Pclass=3的存活几率非常低
    2、Pclass=2和3中，年龄小的也活下来挺多
    3、Pclass=1的存活几率挺高的
'''

grid2 = sns.FacetGrid(train_df,row = 'Embarked' , size = 2.2 ,aspect = 1.6)
grid2.map(sns.pointplot,'Pclass','Survived','Sex',palette = 'deep')
grid2.add_legend()
'''
观察图像好像可能看出：
    1、除了第二个登录点外，女的生存几率挺高
    2、不同的登录点可能有不同的关联性，所以把这个feature也考虑进去吧
'''


grid3 = sns.FacetGrid(train_df,row = 'Embarked',col = 'Survived' , size=2.2 , aspect = 1.6)
grid3.map(sns.barplot,'Sex','Fare',alpha = 0.5 , ci = None)
grid3.add_legend()
'''
观察图像好像可能看出：
    1、票价高的存活几率大点
    2、好像确实登录点与生存有点关系
'''


# 基于以上的这些观察，对数据集做一些处理，删除一些没用的，创造一些更好的feature
# 注意点就是要保证测试集和训练集同步
train_df = train_df.drop(['Ticket','Cabin'],axis = 1)
test_df =test_df.drop(['Ticket','Cabin'],axis = 1)


# 确定一下names和passengerID是否可以删除
combine = [train_df,test_df]
for dataset in combine:
    #把名字中的Mr等内容根据正则选出来
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(train_df['Title'],train_df['Sex'])

# 替换title的字段，把稀有度较低的结合成一类
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

train_df[['Title','Survived']].groupby(['Title'],as_index=False).mean()

# 把字符串转化成标称数据
title_mapping = {'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)     #处理缺失值

train_df = train_df.drop(['Name','PassengerId'],axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df,test_df]

# 把sex也转化成标称数据把
set_mapping = {'female':1,'male':0}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(set_mapping).astype(int)
        
# 还要处理一下Age中的缺失数据0_0
grid4 = sns.FacetGrid(train_df,row = 'Pclass',col = 'Sex',size = 2.2 , aspect = 1.6)
grid4.map(plt.hist,'Age',alpha = 0.5 , bins = 20)
grid4.add_legend()

# 用Pclass和Sex来随机给出比较好
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0,2):     # 性别只有两个
        for j in range (0,3): #仓位有三个值
            guess_df =dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int(age_guess/0.5+0.5)*0.5 # 数据集中有.5
    for i in range(0,2):
        for j in range (0,3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)

# 把处理好的Age feature 转化成区间属性类型~
train_df['AgeBand'] = pd.cut(train_df['Age'],5)

#看下效果如何
train_df[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='AgeBand',ascending = True)

# 再把区间类型的转化成标称类型的....
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16,'Age'] = 0
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32) ,'Age'] = 1
    dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48) ,'Age'] = 2
    dataset.loc[(dataset['Age']>48) & (dataset['Age']<=64) ,'Age'] = 3
    dataset.loc[dataset['Age']>64,'Age'] = 4
    
# 删除掉ageband
train_df = train_df.drop(['AgeBand'],axis=1)
combine = [train_df,test_df]

# 把兄弟姐妹数量和父母子女数量结合在一起组成新的变量
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
# 看看数据如何
train_df[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending = True)

# 这个数据再变一下变成是否单身，根据FamilySize ？= 1
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1 , 'IsAlone'] = 1
    
# 看看数据如何
train_df[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean()

# 删掉没用的feature
train_df = train_df.drop(['SibSp','Parch','FamilySize'],axis=1)
test_df = test_df.drop(['SibSp','Parch','FamilySize'],axis=1)
combine = [train_df,test_df]

# 把登录口的字符串数据缺失值填上，并且变成数值型
freq_port = train_df.Embarked.dropna()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
    
# 处理一下船票价格的缺失值
test_df['Fare'].fillna(test_df['Fare'].dropna().median(),inplace = True)

# 把船票价格的类型转化为标称型
train_df['FareBand'] = pd.qcut(train_df['Fare'],4)
train_df[['FareBand','Survived']].groupby(['FareBand'],as_index=False).mean().sort_values(by='FareBand',ascending=True)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop(['FareBand'],axis=1)
combine = [train_df,test_df]

# 模型环节啦~
X_train = train_df.drop('Survived',axis = 1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId',axis = 1).copy()

# 划分一下测试集和训练集,用训练集中的部分来训练，用测试集来看效果
trainX,testX,trainY,testY = train_test_split(X_train,Y_train,train_size = 0.7)

# logistic regression
logreg = LogisticRegression()     #有很多参数等会调一下...
logreg.fit(trainX,trainY)
logreg.score(testX,testY)   #0.7873
proba_logistic = logreg.predict_proba(testX)[:,1]
precision_logistic,recall_logistic,thresholds_logistic = precision_recall_curve(testY,proba_logistic)
plt.plot(recall_logistic,precision_logistic,label ='logistic')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
print (classification_report(testY,logreg.predict(testX)))     #看一下F1值  0.79
print (roc_auc_score(testY,proba_logistic))     # 看一下auc的面积 0.8587

param_logreg = {'C':[0.001,0.01,0.1,1,10],'max_iter':[100,250]}
clf = GridSearchCV(logreg,param_logreg,cv = 5,n_jobs = -1,verbose = 1 ,scoring = 'roc_auc')
clf.fit(trainX,trainY)
clf.grid_scores_
clf.best_params_       # 看一下最佳组合C=0.1 max_iter=10，优化一下模型

logreg = LogisticRegression(C = 0.1,max_iter = 100) # 我擦，怎么全面下降了....!!!
logreg.fit(trainX,trainY)
logreg.score(testX,testY)           # 0.7873
proba_logistic = logreg.predict_proba(testX)[:,1]
precision_logistic,recall_logistic,thresholds_logistic = precision_recall_curve(testY,proba_logistic)
plt.plot(recall_logistic,precision_logistic,label ='logistic')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
print (classification_report(testY,logreg.predict(testX)))        #0.79
print (roc_auc_score(testY,proba_logistic))         # 0.8571

# SVM
svc = SVC(probability=True)
svc.fit(trainX,trainY)
svc.score(trainX,trainY)      # 0.8459
proba_svc = svc.predict_proba(testX)[:,1]
precision_svc,recall_svc,thresholds_svc = precision_recall_curve(testY,proba_svc)
plt.plot(recall_svc,precision_svc,label ='svc')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
print (classification_report(testY,svc.predict(testX)))     #看一下F1值  0.81
print (roc_auc_score(testY,proba_svc))     # 看一下auc的面积  0.8291

param_svc = {'C':[0.001,0.01,0.1,1,10],'cache_size':[150,200,250],'max_iter':[100,150,250]}
clf = GridSearchCV(svc,param_svc,cv = 5,n_jobs = -1,verbose = 1 ,scoring = 'roc_auc')
clf.fit(trainX,trainY)
clf.grid_scores_
clf.best_params_  

svc = SVC(probability = True,C = 1,cache_size = 150 , max_iter = 150)
svc.fit(trainX,trainY)
svc.score(trainX,trainY)      # 0.8459
proba_svc = svc.predict_proba(testX)[:,1]
precision_svc,recall_svc,thresholds_svc = precision_recall_curve(testY,proba_svc)
plt.plot(recall_svc,precision_svc,label ='svc')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
print (classification_report(testY,svc.predict(testX)))     #看一下F1值  0.81
print (roc_auc_score(testY,proba_svc))       # 0.8376

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(trainX,trainY)
knn.score(trainX,trainY)  # 0.8635
proba_knn = knn.predict_proba(testX)[:,1]
precision_knn,recall_knn,thresholds_knn = precision_recall_curve(testY,proba_knn)
plt.plot(recall_knn,precision_knn,label ='knn')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
print (classification_report(testY,knn.predict(testX)))     #看一下F1值  0.77
print (roc_auc_score(testY,proba_knn))      # 0.8138

param_knn = {'n_neighbors':[1,3,5,7,9],'leaf_size':[20,25,35],'p':[1,2,3]}
clf = GridSearchCV(knn,param_knn,cv = 5,n_jobs = -1,verbose = 1 ,scoring = 'roc_auc')
clf.fit(trainX,trainY)
clf.grid_scores_
clf.best_params_   #{'leaf_size': 35, 'n_neighbors': 7, 'p': 2}

knn = KNeighborsClassifier(n_neighbors=7 , leaf_size = 35 , p = 2)
knn.fit(trainX,trainY)
knn.score(trainX,trainY)  # 0.8507
proba_knn = knn.predict_proba(testX)[:,1]
precision_knn,recall_knn,thresholds_knn = precision_recall_curve(testY,proba_knn)
plt.plot(recall_knn,precision_knn,label ='knn')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
print (classification_report(testY,knn.predict(testX)))     #看一下F1值  0.77
print (roc_auc_score(testY,proba_knn))      # 0.8304


# 朴素贝叶斯
bayes = GaussianNB()
bayes.fit(trainX,trainY)
bayes.score(trainX,trainY)        #0.7849
proba_bayes = bayes.predict_proba(testX)[:,1]
precision_bayes,recall_bayes,thresholds_bayes = precision_recall_curve(testY,proba_bayes)
plt.plot(recall_bayes,precision_bayes,label ='bayes')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
print (classification_report(testY,bayes.predict(testX)))     #看一下F1值  0.78
print (roc_auc_score(testY,proba_bayes))  # 0.8430
# 贝叶斯没有参数...


# 感知器
perceptron = Perceptron()
perceptron.fit(trainX,trainY)
perceptron.score(trainX,trainY)     # 0.7592
# 感知器没有那些评价标准貌似9_9
 
param_perceptron = {'alpha':[0.0001,0.00005,0.00001,0.000005],'n_iter':[1,3,5]}
clf = GridSearchCV(perceptron,param_perceptron,cv = 5,n_jobs = -1,verbose = 1 ,scoring = 'roc_auc')
clf.fit(trainX,trainY)
clf.grid_scores_
clf.best_params_   # {'alpha': 0.0001, 'n_iter': 1}

perceptron = Perceptron(alpha = 0.0001, n_iter = 1)
perceptron.fit(trainX,trainY)
perceptron.score(trainX,trainY)     # 0.6597


# 线性SVC
linear_svc = LinearSVC()
linear_svc.fit(trainX,trainY)
linear_svc.score(trainX,trainY)    # 0.7977
# 线性SVC没有那些评价标准貌似9_9

param_linear_svc = {'C':[0.01,0.1,1,5,10],'max_iter':[500,1000,1500]}
clf = GridSearchCV(linear_svc,param_linear_svc,cv = 5,n_jobs = -1,verbose = 1 ,scoring = 'roc_auc')
clf.fit(trainX,trainY)
clf.grid_scores_
clf.best_params_   # {'C': 0.1, 'max_iter': 500}

linear_svc = LinearSVC(C = 0.1 , max_iter = 500)
linear_svc.fit(trainX,trainY)
linear_svc.score(trainX,trainY)    # 0.7994


# 随机梯度下降SGD-stochastic gradient descent
sgd = SGDClassifier(loss = 'log')
sgd.fit(trainX,trainY)
sgd.score(trainX,trainY)       # 0.8057
proba_sgd = sgd.predict_proba(testX)[:,1]
precision_sgd,recall_sgd,thresholds_sgd = precision_recall_curve(testY,proba_sgd)
plt.plot(recall_sgd,precision_sgd,label ='sgd')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
print (classification_report(testY,sgd.predict(testX)))     #看一下F1值  0.78
print (roc_auc_score(testY,proba_sgd))       # 0.8578

param_sgd = {'n_iter':[1,3,5,10,20],'verbose':[1,2,3,4,5]}
clf = GridSearchCV(sgd,param_sgd,cv = 5,n_jobs = -1,verbose = 1 ,scoring = 'roc_auc')
clf.fit(trainX,trainY)
clf.grid_scores_
clf.best_params_    # {'n_iter': 3, 'verbose': 1}

sgd = SGDClassifier(loss = 'log' , n_iter = 3 , verbose = 2)
sgd.fit(trainX,trainY)
sgd.score(trainX,trainY)       # 0.7913
proba_sgd = sgd.predict_proba(testX)[:,1]
precision_sgd,recall_sgd,thresholds_sgd = precision_recall_curve(testY,proba_sgd)
plt.plot(recall_sgd,precision_sgd,label ='sgd')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
print (classification_report(testY,sgd.predict(testX)))     #看一下F1值  0.78
print (roc_auc_score(testY,proba_sgd))  #0.8457


# 决策树 decision tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(trainX,trainY)
decision_tree.score(trainX,trainY)       # 0.8796有点高啊...
proba_decision_tree = decision_tree.predict_proba(testX)[:,1]
precision_decision_tree,recall_decision_tree,thresholds_decision_tree = precision_recall_curve(testY,proba_decision_tree)
plt.plot(recall_decision_tree,precision_decision_tree,label ='decision_tree')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
print (classification_report(testY,decision_tree.predict(testX)))     # 看一下F1值  0.75
print (roc_auc_score(testY,proba_decision_tree))    # 0.7569

param_decision_tree = {'criterion':['gini','entropy'],'max_features':['auto','sqrt','log2']}
clf = GridSearchCV(decision_tree,param_decision_tree,cv = 5,n_jobs = -1,verbose = 1 ,scoring = 'roc_auc')
clf.fit(trainX,trainY)
clf.grid_scores_
clf.best_params_    # {'criterion': 'entropy', 'max_features': 'auto'}

decision_tree = DecisionTreeClassifier(criterion = 'entropy' , max_features = 'auto')
decision_tree.fit(trainX,trainY)
decision_tree.score(trainX,trainY)       # 0.8796有点高啊...
proba_decision_tree = decision_tree.predict_proba(testX)[:,1]
precision_decision_tree,recall_decision_tree,thresholds_decision_tree = precision_recall_curve(testY,proba_decision_tree)
plt.plot(recall_decision_tree,precision_decision_tree,label ='decision_tree')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
print (classification_report(testY,decision_tree.predict(testX)))     # 看一下F1值  0.73
print (roc_auc_score(testY,proba_decision_tree))  # 0.7449


# 随机森林 random forest
random_forest = RandomForestClassifier()
random_forest.fit(trainX,trainY)
random_forest.score(trainX,trainY)       # 0.8748
proba_random_forest = random_forest.predict_proba(testX)[:,1]
precision_random_forest,recall_random_forest,thresholds_random_forest = precision_recall_curve(testY,proba_random_forest)
plt.plot(recall_random_forest,precision_random_forest,label ='random_forest')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
print (classification_report(testY,random_forest.predict(testX)))     # 看一下F1值  0.79
print (roc_auc_score(testY,proba_random_forest))     # 0.7943


param_random_forest = {'n_estimators':[5,10,15,20]}
clf = GridSearchCV(random_forest,param_random_forest,cv = 5,n_jobs = -1,verbose = 1 ,scoring = 'roc_auc')
clf.fit(trainX,trainY)
clf.grid_scores_
clf.best_params_    # {'n_estimators': 15}

random_forest = RandomForestClassifier(n_estimators = 15)
random_forest.fit(trainX,trainY)
random_forest.score(trainX,trainY)       # 0.8796
proba_random_forest = random_forest.predict_proba(testX)[:,1]
precision_random_forest,recall_random_forest,thresholds_random_forest = precision_recall_curve(testY,proba_random_forest)
plt.plot(recall_random_forest,precision_random_forest,label ='random_forest')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
print (classification_report(testY,random_forest.predict(testX)))     # 看一下F1值  0.76
print (roc_auc_score(testY,proba_random_forest))     # 0.7944

# 因为上面的参数都是我瞎写的，毕竟我还不是很了解这些参数的意义，所以，我们来决定下哪下用默认的，那些调整
# 模型的评估（model evaluation）
# 因为比之前增加了两个参数，一个是f1值，一个是roc的面积，所以一起看看吧

models_data = pd.DataFrame(
        {
        'Score':[svc.score(trainX,trainY),knn.score(trainX,trainY),logreg.score(testX,testY),
                random_forest.score(trainX,trainY),bayes.score(trainX,trainY),perceptron.score(trainX,trainY),
                sgd.score(trainX,trainY),linear_svc.score(trainX,trainY),decision_tree.score(trainX,trainY)],
        'Roc-auc':[roc_auc_score(testY,proba_svc),roc_auc_score(testY,proba_knn),roc_auc_score(testY,proba_logistic),roc_auc_score(testY,proba_random_forest),roc_auc_score(testY,proba_bayes),np.nan,
            roc_auc_score(testY,proba_sgd),np.nan,roc_auc_score(testY,proba_decision_tree)],
        },
    index = ['Support Vector Machines','KNN','Logistic Regression','Random Forest','Naive Bayes','Perceptron','Stochastic Gradient Decent','Linear SVC','Decision Tree'])

models_data.sort_values(by='Roc-auc',ascending = False)

X_train = train_df.drop('Survived',axis = 1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId',axis = 1).copy()

submission = pd.DataFrame({
        'PassengerId':test_df['PassengerId'],
        'Survived':logreg.predict(X_test)
        })

submission.to_csv('C:/Users/carne/Desktop/submission.csv',index=False)




