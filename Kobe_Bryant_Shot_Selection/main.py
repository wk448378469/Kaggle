# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 11:19:32 2017

@author: 凯风
"""

'''
    数据说明：
        数据集无缺失数据~~~
        
    字段说明：
        action_type                 动作类型                                over
        combined_shot_type          投篮姿势                                over
        game_event_id               投篮在比赛中的发生事件ID                   删掉
        game_id                     比赛ID                                over
        lat                         投篮定位1                               删掉
        loc_x                       投篮定位2                               over
        loc_y                       投篮定位3                               over
        lon                         投篮定位4                               删掉
        minutes_remaining           投篮时距离本节结束分钟，0-11                over
        period                      投篮时处于比赛的第几小节，1-7            over
        playoffs                    是否是季后赛                          over
        season                      第几赛季(一共二十个赛季)               over
        seconds_remaining           投篮时距离本节结束秒数，0-59            over
        shot_distance               投篮距离                                over
        shot_made_flag              是否得分 
        shot_type                   两分球还是三分球                        over
        shot_zone_area              投篮区域1                               over
        shot_zone_basic             投篮区域2                               over
        shot_zone_range             投篮区域3                               over
        team_id                     所在队伍ID                            删掉
        team_name                   所在队伍名称                          删掉
        game_date                   比赛日期                             over
        matchup                     比赛对阵双方，@-表示 vs-表示          over
        opponent                    比赛对手                             over
        shot_id                     投篮ID
'''

def process_part1(data):
    
    # 将combined_shot_type这个分类特征使用one-hot-encode
    dummy_combined_shot_type = pd.get_dummies(data['combined_shot_type'],prefix='cstype')
    data = data.join(dummy_combined_shot_type)
    
    # 将shot_type这个分类特征使用one-hot-encode
    dummy_shot_type = pd.get_dummies(data['shot_type'],prefix='stype')
    data = data.join(dummy_shot_type)
    
    # action_type，将很多不常用的动作合并到一起
    action_type = data.action_type.value_counts()       # 数量太多了，57个，所以处理一下
    mostActionType = list(action_type[:15].index)        # 前14个动作,覆盖了95以上的数据
    data['action_type'] = data['action_type'].apply(lambda x: x if x in mostActionType else 'others')
    
    # 将action_type这个分类特征使用one-hot-encode
    dummy_action_type = pd.get_dummies(data['action_type'],prefix='atype')
    data = data.join(dummy_action_type)

    # 计算两分球，三分球的命中率
    totalTwoBalls = data[data['shot_made_flag'] == 1].groupby(['shot_type']).sum()['shot_made_flag'][0]
    totalThreeBalls = data[data['shot_made_flag'] == 1].groupby(['shot_type']).sum()['shot_made_flag'][1]
    twoBallShoting = totalTwoBalls/data['stype_2PT Field Goal'].sum()
    threeBallShoting = totalThreeBalls/data['stype_3PT Field Goal'].sum()

    data['ballShoting'] = data['shot_type'].apply(lambda x:twoBallShoting if '2PT' in x else threeBallShoting)
    
    # 计算不同combined_shot_type的命中率
    dic_totalnum = dict(data[data['shot_made_flag'].isnull() == False].combined_shot_type.value_counts())
    dic_shot = dict(data[data['shot_made_flag'] == 1].combined_shot_type.value_counts())
    dic_shoting = {}
    for key in dic_shot:
        shotingValue = dic_shot[key] / dic_totalnum[key]
        dic_shoting[key] = shotingValue
    data['typeShoting'] = data['combined_shot_type'].map(dic_shoting)
    
    # 删除掉没用的feature
    data.drop(['combined_shot_type','shot_type','action_type'],axis=1,inplace=True)
    
    return data

def process_part2(data):
    # lat , loc_x , loc_y , lon                             数值型
    # shot_zone_area ，shot_zone_basic ，shot_zone_range    标称型
    # shot_distance                                         数值型

    # 极坐标
    loc_x_zero = data['loc_x'] == 0
    data['angle'] = np.array([0] * len(data))
    data['angle'][~loc_x_zero] = np.arctan(data['loc_y'][~loc_x_zero] / data['loc_x'][~loc_x_zero])
    data['angle'][loc_x_zero] = np.pi / 2

    # 计算每个shot_zone_basic的命中率
    zone_totalnum = dict(data[data['shot_made_flag'].isnull() == False].shot_zone_basic.value_counts())
    zone_shot = dict(data[data['shot_made_flag'] == 1].shot_zone_basic.value_counts())
    zone_shoting = {}
    for key in zone_shot:
        shotingValue = zone_shot[key] / zone_totalnum[key]
        zone_shoting[key] = shotingValue
    data['zoneShoting'] = data['shot_zone_basic'].map(zone_shoting)
    
    # one hot encode
    data['shot_distance_group'] = pd.cut(data['shot_distance'],7)
    dic = ['shot_zone_area','shot_zone_basic','shot_zone_range','shot_distance_group']
    for feature in dic:
        dummy = pd.get_dummies(data[feature])
        dummy = dummy.add_prefix('{}#'.format(feature))
        data.drop(feature,axis=1,inplace=True)
        data = data.join(dummy)
    
    return data

def process_part3(data):
    # seconds_remaining,minutes_remaining       投篮点剩余时间
    # game_date                                 比赛的日期
    # period                                    投篮处于的小节数
    data['timeStart'] = 60 * (11 - data['minutes_remaining']) + (60 - data['seconds_remaining'])
    data['timeRemain'] = 60 * data['minutes_remaining'] + data['seconds_remaining']
    data['timeGameStart'] = (data['period'] <= 4).astype(int)*(data['period']-1)*12*60 + (data['period'] > 4).astype(int)*((data['period']-4)*5*60 + 3*12*60) + data['timeStart']
    data['isLastTime'] = data['timeRemain'].apply(lambda x: 1 if x < 10 else 0 )
    
    data['game_date'] = pd.to_datetime(data['game_date'])
    data['year'] = data['game_date'].dt.year
    data['month'] = data['game_date'].dt.month
    data['day'] = data['game_date'].dt.day
    data['dayofweek'] = data['game_date'].dt.dayofweek
    data['dayofyear'] = data['game_date'].dt.dayofyear
    
    dummy = pd.get_dummies(data['period'],prefix='period')
    data = data.join(dummy)
    data['isRegularTime'] = data.apply(lambda row: 1 if row['period'] <= 4 else 0 ,axis=1)
    data['isOverTime'] = data.apply(lambda row: 1 if row['period'] > 4 else 0 ,axis=1)
    
    data.drop(['game_date','minutes_remaining','seconds_remaining','period'],axis=1,inplace=True)
    return data

def process_part4(data):
    # game_id                       比赛ID，计算比赛的命中率
    # matchup                       对阵信息，是主场还是客场
    # opponent                      比赛对手，onehot
    # playoffs                      是否是季后赛，新增加一个是否是常规赛
    # season                        赛季，计算赛季命中率
    game_totalnum = dict(data[data['shot_made_flag'].isnull() == False].game_id.value_counts())
    game_shot = dict(data[data['shot_made_flag'] == 1].game_id.value_counts())
    game_shoting = {}
    for key in game_totalnum:
        if key not in game_shot:
            shotingValue = 0
            game_shoting[key] = shotingValue
        else:
            shotingValue = game_shot[key] / game_totalnum[key]
            game_shoting[key] = shotingValue
    data['gameShoting'] = data['game_id'].map(game_shoting)    
    
    data['isHome'] = data['matchup'].apply(lambda x: 1 if 'vs' in str(x) else 0 )
    data['isAway'] = data['matchup'].apply(lambda x: 1 if '@' in str(x) else 0 )
    
    dummy_opponent = pd.get_dummies(data['opponent'],prefix = 'opponent')
    data = data.join(dummy_opponent)
    
    data['isRegular'] = data['playoffs'].apply(lambda x: 1 if x == 0 else 0)
    
    dummy_season = pd.get_dummies(data['season'],prefix = 'season')
    data = data.join(dummy_season)
    
    season_totalnum = dict(data[data['shot_made_flag'].isnull() == False].season.value_counts())
    season_shot = dict(data[data['shot_made_flag'] == 1].season.value_counts())
    season_shoting = {}
    for key in season_totalnum:
        if key not in season_shot:
            shotingValue = 0
            season_shoting[key] = shotingValue
        else:
            shotingValue = season_shot[key] / season_totalnum[key]
            season_shoting[key] = shotingValue
    data['seasonShoting'] = data['season'].map(season_shoting)       
    
    data.drop(['game_id','matchup','opponent','season'],axis=1,inplace=True)

    return data

def process_part5(data,isStandardized=True):
        
    testX = data[data['shot_made_flag'].isnull() == True]
    testY = data['shot_id']
    testX.drop(['shot_made_flag','shot_id'],axis=1,inplace=True)
    testX = testX.fillna(0)
    
    trainX = data[data['shot_made_flag'].isnull() == False]
    trainY = trainX['shot_made_flag']
    trainX.drop(['shot_id','shot_made_flag'],axis=1,inplace=True)
    
    # 标准化数据
    if isStandardized == True:
        from sklearn.preprocessing import StandardScaler
        SS = StandardScaler()
        SS.fit(trainX)
        trainX = SS.transform(trainX)
        testX = SS.transform(testX)
    
    # pca降维
    from sklearn.decomposition import PCA
    varianceEnough = False
    n_com = 50
    while not varianceEnough:
        pca = PCA(n_components=n_com)
        pca.fit(trainX,trainY)
        n_com = n_com + 1
        if np.sum(pca.explained_variance_ratio_) > 0.9:
            varianceEnough = True
            
    trainX = pca.transform(trainX)
    testX = pca.transform(testX)
    



if __name__ == '__main__':
    # 读取数据
    import pandas as pd
    import numpy as np
    data = pd.read_csv('D:/mygit/Kaggle/Kobe_Bryant_Shot_Selection/data.csv')
    data.drop(['team_id','team_name','lat','lon','game_event_id'],axis=1,inplace=True)
    
    # 处理数据~~~
    # 主要处理投篮动作相关的特征
    data = process_part1(data)
    
    # 处理投篮点相关的特征
    data = process_part2(data)
    
    # 处理时间相关的特征
    data = process_part3(data)
    
    # 处理比赛相关的特征
    data = process_part4(data)
    
    # 标准化、pca、分割数据、保存数据~
    data = process_part5(data)
    
    