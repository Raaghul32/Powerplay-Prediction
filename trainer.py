# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 12:19:43 2021

@author: raagh
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



train = pd.read_csv('all_matches.csv')
#only taking last 4 years into account
train['season'] = train['season'].astype(str)
train = train[train['season'].isin(['2021','2020/21','2019','2018'])]
train = train[train['innings'].isin([1,2])]
train['over'],train['ball_in_over'] =  divmod(train['ball'],1)
train['over'] = train['over'] + 1
train = train[train['over']<=6]
train['Total Runs'] = train['runs_off_bat'] + train['extras']
grouped_train = train.groupby(['match_id','batting_team','bowling_team','venue','innings']).agg({'Total Runs' : 'sum'})
striker = train.groupby(['match_id','batting_team','bowling_team','venue','innings'])['striker'].apply(set)
bowler = train.groupby(['match_id','batting_team','bowling_team','venue','innings'])['bowler'].apply(set)
sr = striker.to_frame()
bat_list = []
sr = sr.reset_index()
for i in range(len(sr)):
    a_set = sr['striker'][i]
    list_of_strings = [str(s) for s in a_set]
    joined_string = ",".join(list_of_strings)
    bat_list.append(joined_string)
sr['Bat'] = bat_list
bowl_list =[]
bw = bowler.to_frame()
bw = bw.reset_index()
for i in range(len(bw)):
    a_set = bw['bowler'][i]
    list_of_strings = [str(s) for s in a_set]
    joined_string = ",".join(list_of_strings)
    bowl_list.append(joined_string)
bw['Bowl'] = bowl_list
sr = sr[['match_id','innings','Bat']]
bw = bw[['match_id','innings','Bowl']]
runs = grouped_train.reset_index()
train = runs.merge(sr,on = ['match_id','innings']).merge(bw,on = ['match_id','innings'])
train_copy = train.copy()
train_copy.replace('Delhi Daredevils','Delhi Capitals',inplace = True)
train_copy.replace('Kings XI Punjab','Punjab Kings',inplace = True)
train_copy.replace('MA Chidambaram Stadium, Chepauk, Chennai','MA Chidambaram Stadium',inplace = True)
train_copy.replace('Wankhede Stadium, Mumbai','Wankhede Stadium',inplace = True)
def batbowlrank(data) :
    ranked_bat = data.groupby(['batting_team'],as_index = False).agg({'Total Runs':'mean'})
    ranked_bat.columns = ['Team','Bat Runs']
    ranked_bowl = data.groupby(['bowling_team'],as_index = False).agg({'Total Runs':'mean'})
    ranked_bowl.columns = ['Team','Bowl Runs']
    ranked = ranked_bat.merge(ranked_bowl,on = 'Team' )
    ranked['Bat Rank'] = ranked['Bat Runs'].rank(ascending = True)
    ranked['Bowl Rank'] = ranked['Bowl Runs'].rank(ascending = False)
    return ranked
ranked_op = batbowlrank(train_copy)
train_copy['venue'].value_counts()
venues = ['MA Chidambaram Stadium','Wankhede Stadium','Eden Gardens','Arun Jaitley Stadium','M.Chinnaswamy Stadium','Narendra Modi Stadium']   
train_copy['venue_update'] = np.where(train_copy['venue'].isin(venues),train_copy['venue'],'others')
ranked_stadium = train_copy.groupby('venue_update',as_index = False).agg({'Total Runs':'mean'})
ranked_stadium['Venue Rank'] = ranked_stadium['Total Runs'].rank(ascending = True)
ven_rank = ranked_stadium[['venue_update','Venue Rank']].set_index('venue_update').to_dict()
bat_rank = ranked_op[['Team','Bat Rank']].set_index('Team').to_dict('dict')
bowl_rank = ranked_op[['Team','Bowl Rank']].set_index('Team').to_dict()

def get_number(data):
    bat_len = []
    bowl_len = []

    for i in range(len(data)):

        bat_len.append(len(data['Bat'][i].split(',')))
        bowl_len.append(len(data['Bowl'][i].split(',')))
    return bat_len,bowl_len
 
train_copy['Bat Number'],train_copy['Bowl Number'] = get_number(train_copy)
train_copy['battin_team_no'] = train_copy['batting_team'].map(bat_rank['Bat Rank'])
train_copy['bowling_team_no'] = train_copy['bowling_team'].map(bowl_rank['Bowl Rank'])
train_copy['venue_no'] = train_copy['venue_update'].map(ven_rank['Venue Rank'])

train_data = train_copy[['battin_team_no','bowling_team_no','innings','Bat Number','Bowl Number','Total Runs']]
train_data.to_csv('submissionFormat/trainer.csv',index = False)
x = train_data.iloc[:,:-1].values
y = train_data.iloc[:,-1].values

f,ax = plt.subplots(figsize=(10, 8))
corr = train_data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)





x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20)
x_train = sc.fit_transform(x_train)
x_test  = sc.transform(x_test )
accuracy = []
names = []
rmse = []





################################################################


#using neural networks






from keras.models import Sequential
from keras.layers import Dense



# define the model
model = Sequential()
model.add(Dense(13, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='relu'))

model.add(Dense(3,kernel_initializer = 'normal',activation = 'relu'))

model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train,y_train,epochs=200)
model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)

import joblib
filename = 'submissionFormat/model.joblib'
joblib.dump(model,filename)

model.save('submissionFormat/my_nn_model.h5')  # creates a HDF5 file 'my_model.h5'


