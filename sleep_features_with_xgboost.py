import pandas as pd
import numpy as np

from scipy.stats import zscore

from sklearn.preprocessing import MinMaxScaler
#from yellowbrick.classifier import ClassPredictionError
from sklearn.metrics import accuracy_score

sleep_features_path= '/nesi/project/aut03802/Data/sleep_features/'
features_csv='/nesi/project/aut03802/Data/features.csv'


df_features=pd.read_csv(features_csv)
print("Total Data:",len(df_features))
print("Missing Data:",df_features.isnull().any(axis=1).sum())
print("Actual Instances of Data:",len(df_features)-df_features.isnull().any(axis=1).sum())

df_features.drop(['Unnamed: 0.1', 'Unnamed: 0'],axis=1, inplace=True)

#merge sleep features for each participant
num_of_participants=16
for i in range(num_of_participants):
    if i<9:
        participant_id='00'+str(i+1)
    else:
        participant_id='0'+str(i+1)
    sleep_end_points_p1=sleep_features_path+participant_id+'/sleep_endpoints/sleep_endpoints_summary.csv'
    sleep_df=pd.read_csv(sleep_end_points_p1)
    sleep_df['day_of_impact']=sleep_df['day']+1
    df_participant = df_features[df_features['Participant'] == i+1]
    df_participant['day'] = df_participant['Date'].rank(method='dense').astype(int)
    df_participant['day_of_impact']=df_participant['day']
    merged_df = df_participant.merge(sleep_df, on='day_of_impact' , how='outer')
    if i==0:
        
        new_df=merged_df
    else:
        new_df=pd.concat([merged_df,new_df])
        

    
new_df =new_df.dropna()

numeric_features=['HR_Mean',
                  'HR_Max',
                  'HR_Std',
                  'HR_Skew',
                  'HR_Q1G', 
                  'HR_Q3G', 
                  'EDA_Mean',
       'EDA_Max', 'EDA_Std', 'EDA_Skew', 'EDA_Q1G', 'EDA_Q3G', 'TEMP_Mean',
       'TEMP_Max', 'TEMP_Std', 'TEMP_Skew', 'TEMP_Q1G', 'TEMP_Q3G', 'ACC_Mean',
       'ACC_Max', 'ACC_Std', 'ACC_Skew', 'ACC_Q1G', 'ACC_Q3G', 'BVP_Mean',
       'BVP_Max', 'BVP_Std', 'BVP_Skew', 'BVP_Q1G', 'BVP_Q3G', 'HbA1c', 'PeakEDA', 'PeakEDA_2hrsum', 'PeakEDA_2hrmean', 'maxHRV',
       'minHRV', 'medianHRV', 'meanHRV', 'SDNN', 'NN50', 'pNN50', 'RMSSD',
       'calories2hr', 'protien2hr', 'sugar2hr', 'carbs2hr', 'calories8r',
       'protien8hr', 'sugar8hr', 'carbs8hr', 'calories24hr', 'protien24hr',
       'sugar24hr', 'carbs24hr', 'Eat', 'Eatcnt2hr', 'Eatcnt8hr', 'Eatcnt24hr',
       'Eatmean2hr', 'Eatmean8hr', 'Eatmean24hr', 'WakeTime', 'Minfrommid',
       'Hourfrommid', 'EDA_Min', 'calories8hr', 'TEMP_Min', 'BVP_Min',
       'ACC_Min', 'HR_Min', 'Wake_Time', 'HR_Mean_Historical',
       'HR_Std_Historical', 'ACC_Mean_Historical', 'ACC_Std_Historical',
       'Activity_Bouts', 'Activity24','Activity1h']

categorical_features = ['Participant','Gender']

new_df.to_csv('/nesi/project/aut03802/Data/sleep_plus_features.csv')



