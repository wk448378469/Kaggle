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
        game_event_id               投篮在比赛中的发生事件ID               生产一个新的特征，记为进程选择？
        game_id                     比赛ID                                根据比赛ID计算每场比赛的命中率
        lat                         投篮定位1
        loc_x                       投篮定位2
        loc_y                       投篮定位3
        lon                         投篮定位4
        minutes_remaining           投篮时距离本节结束分钟，0-11
        period                      投篮时处于比赛的第几小节，1-7
        playoffs                    是否是季后赛                          不需要处理
        season                      第几赛季(一共二十个赛季)               将生涯分成不同的阶段？太主观了吧...
        seconds_remaining           投篮时距离本节结束秒数，0-59            和minutes_remaining一起处理
        shot_distance               投篮距离
        shot_made_flag              是否得分
        shot_type                   两分球还是三分球
        shot_zone_area              投篮区域1             
        shot_zone_basic             投篮区域2
        shot_zone_range             投篮区域3
        team_id                     所在队伍ID                            删掉
        team_name                   所在队伍名称                          删掉
        game_date                   比赛日期                             这是时间序列，当然也可以生成是否是背靠背的feature
        matchup                     比赛对阵双方，@-表示 vs-表示           生产主客场特征
        opponent                    比赛对手                            对手生涯命中率？
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
    def smallType(x):
        if x not in mostActionType:
            x = 'others'
        return x
    data['action_type'] = data['action_type'].apply(smallType)
    
    # 将action_type这个分类特征使用one-hot-encode
    dummy_action_type = pd.get_dummies(data['action_type'],prefix='atype')
    data = data.join(dummy_action_type)

    # 计算两分球，三分球的命中率
    totalTwoBalls = data[data['shot_made_flag'] == 1].groupby(['shot_type']).sum()['shot_made_flag'][0]
    totalThreeBalls = data[data['shot_made_flag'] == 1].groupby(['shot_type']).sum()['shot_made_flag'][1]
    twoBallShoting = totalTwoBalls/data['stype_2PT Field Goal'].sum()
    threeBallShoting = totalThreeBalls/data['stype_3PT Field Goal'].sum()

    data['BallShoting'] = data['shot_type'].apply(lambda x:twoBallShoting if '2PT' in x else threeBallShoting)
    
    # 计算不同combined_shot_type的命中率
    dic_totalnum = dict(data[data['shot_made_flag'].isnull() == False].combined_shot_type.value_counts())
    dic_shot = dict(data[data['shot_made_flag'] == 1].combined_shot_type.value_counts())
    dic_shoting = {}
    for key in dic_shot:
        shotingValue = dic_shot[key] / dic_totalnum[key]
        dic_shoting[key] = shotingValue
    data['TypeShoting'] = data['combined_shot_type'].map(dic_shoting)
    
    # 删除掉没用的feature
    data.drop(['combined_shot_type','shot_type','action_type'],axis=1,inplace=True)
    
    return data

def process_part2(data):
    # lat , loc_x , loc_y , lon                             数值型
    # shot_zone_area ，shot_zone_basic ，shot_zone_range    标称型
    # shot_distance                                         数值型

    # 极坐标
    
    # one hot encode
               
    # 计算每个shot_zone_basic的命中率
    
    # 删除没用的feature
    return data

def process_part3(data):
    return data

def process_part4(data):
    return data

def process_part5(data):
    return data


if __name__ == '__main__':
    # 读取数据
    import pandas as pd
    data = pd.read_csv('D:/mygit/Kaggle/Kobe_Bryant_Shot_Selection/data.csv')
    data.drop(['team_id','team_name'],axis=1,inplace=True)
    
    # 处理数据~~~
    # 主要处理投篮动作相关的特征
    data = process_part1(data)
    
    # 处理投篮点相关的特征
    data = process_part2(data)
    
    # 处理时间相关的特征
    data = process_part3(data)
    
    # 处理比赛相关的特征
    data = process_part4(data)
    
    # 其他
    data = process_part5(data)
    
    # 分割数据集
    test = data[data['shot_made_flag'].isnull() == True]
    train = data[data['shot_made_flag'].isnull() == False]
    