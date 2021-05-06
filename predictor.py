### Custom definitions and classes if any ###
from keras.models import load_model
import joblib
import pandas as pd

def get_number(data):
    bat_len = []
    bowl_len = []

    for i in range(len(data)):

        bat_len.append(len(data['batsmen'][i].split(',')))
        bowl_len.append(len(data['bowlers'][i].split(',')))
    return bat_len,bowl_len
def predictRuns(testInput):
    
    prediction = 0
    ### Your Code Here ###
    train_copy =  pd.read_csv(testInput)
    bat_rank = {'Chennai Super Kings': 1.0, 'Delhi Capitals': 2.0, 'Kolkata Knight Riders': 3.0, 'Mumbai Indians': 6.0, 'Punjab Kings': 5.0, 'Rajasthan Royals': 4.0, 'Royal Challengers Bangalore': 7.0, 'Sunrisers Hyderabad': 8.0}
    bowl_rank = {'Chennai Super Kings': 6.0, 'Delhi Capitals': 2.0, 'Kolkata Knight Riders': 3.0, 'Mumbai Indians': 7.0, 'Punjab Kings': 1.0, 'Rajasthan Royals': 4.0, 'Royal Challengers Bangalore': 5.0, 'Sunrisers Hyderabad': 8.0}
    #ranks based on performances in the last three years
    train_copy['Bat Number'],train_copy['Bowl Number'] = get_number(train_copy)
    train_copy['battin_team_no'] = train_copy['batting_team'].map(bat_rank)
    train_copy['bowling_team_no'] = train_copy['bowling_team'].map(bowl_rank)
    train_data = train_copy[['battin_team_no','bowling_team_no','innings','Bat Number','Bowl Number']]
    x = train_data.iloc[:,:,].values
    scaler = joblib.load('scaled.joblib')
    model = load_model('my_nn_model.h5')
    model_input = scaler.transform(x)
    prediction = model.predict(model_input)
    return round(prediction[0,0])
