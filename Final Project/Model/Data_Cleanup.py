import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import csv
import pathlib
import typing
from typing import cast

parent_dir = pathlib.Path(__file__).parent
Data_dir = parent_dir / 'Data'

def cleanup_1():
    '''
    1. Add TOTAL_LP column to data
    2. Drop summonerName column
    3. Fill NaN values with -1 for numeric columns
    4. Save updated data to Updated_Data.csv
    '''
    data_path = Data_dir / 'CSE2600_Final_Data.csv'
    data = pd.read_csv(data_path, low_memory=False)
    data['TOTAL_LP'] = 0

    for index, row in data.iterrows():
        tier: int = row['tier_int'] - 1
        rank: int = row['rank_int'] - 1
        lp: int = row['leaguePoints']
        total_lp = lp + (rank * 100) + (tier * 400)
        if total_lp > 2800:
            total_lp = 2800
        index = cast(int, index)
        data.at[index, 'TOTAL_LP'] = total_lp
        
    data.drop(columns=['summonerName'], inplace=True)

    for column in data.columns:
        if data[column].isnull().any() or data[column].isna().any():
            if data[column].dtype == int or data[column].dtype == float:
                data.fillna({column: -1}, inplace=True)
            else:
                print(column, data[column].dtype)
                
    with open(Data_dir / 'Updated_Data.csv', mode='w', encoding='utf-8') as file:
        data.to_csv(file, index=False)


def cleanup_2():
    '''
    1. Remove rows with role_density < 0.5
    2. Remove rows with TOTAL_LP < 10
    3. Save cleaned data to Updated_Data.csv
    '''
    data_path = Data_dir / 'Updated_Data.csv'
    data = pd.read_csv(data_path, low_memory=False)
    
    for index, row in data.iterrows():
        if row['role_density'] < 0.5:
            index = cast(int, index)
            data.drop(index=index, inplace=True)
            
        elif row['TOTAL_LP'] < 10:
            index = cast(int, index)
            data.drop(index=index, inplace=True)
    
    with open(Data_dir / 'Updated_Data.csv', mode='w', encoding='utf-8') as file:
        data.to_csv(file, index=False)

def cleanup_3():
    '''
    1. drop columns with only one unique value
    2. Save cleaned data to Updated_Data.csv
    '''
    data_path = Data_dir / 'Updated_Data.csv'
    data = pd.read_csv(data_path, low_memory=False)
    
    for column in data.columns:
        count = data.value_counts(column).count()
        if count == 1:
            print(f"Dropping column: {column} with {count} unique values")
            data.drop(columns=[column], inplace=True)
    
    with open(Data_dir / 'Updated_Data.csv', mode='w', encoding='utf-8') as file:
        data.to_csv(file, index=False)

def cleanup_4():
    '''
    1. Remove rows with gameDuration_s < 900 seconds (15 minutes)
    2. Save cleaned data to Updated_Data.csv
    '''
    data_path = Data_dir / 'Updated_Data.csv'
    data = pd.read_csv(data_path, low_memory=False)
    
    data = data[data['gameDuration_s'] > 900]
    with open(Data_dir / 'Updated_Data.csv', mode='w', encoding='utf-8') as file:
        data.to_csv(file, index=False)

def cleanup_5():
    '''
    1. Drop unnecessary columns
    2. Save cleaned data to Updated_Data.csv
    '''
    data_path = Data_dir / 'Updated_Data.csv'
    data = pd.read_csv(data_path, low_memory=False)
    
    remove = ['wins', 'losses', 'gameCreation_ms', 'target_role_found', 'role_density', 'leagueId', 'baronKills', 'dragonKills', 'challenge_epicMonsterKillsNearEnemyJungler', 'challenge_epicMonsterKillsWithin30SecondsOfSpawn']
    
    data = data.drop(columns=remove)
    with open(Data_dir / 'Updated_Data.csv', mode='w', encoding='utf-8') as file:
        data.to_csv(file, index=False)
        
def cleanup_6():
    '''
    1. Drop identifier columns
    2. Save cleaned data to Updated_Data.csv
    '''
    data_path = Data_dir / 'Updated_Data.csv'
    data = pd.read_csv(data_path, low_memory=False)
    
    remove = ['puuid', 'riotId', 'matchId', 'participantId', 'championId', 'summonerId', 'teamId', 'riotIdGameName', 'riotIdTagline',
              'summonerLevel', 'matches_examined', ]
    
    for feature in remove:
        if feature in data.columns:
            data = data.drop(columns=[feature])
    
    with open(Data_dir / 'Updated_Data.csv', mode='w', encoding='utf-8') as file:
        data.to_csv(file, index=False)

def cleanup_7():
    '''
    '''
    
    data_path = Data_dir / 'Updated_Data.csv'
    data = pd.read_csv(data_path, low_memory=False)
    
    remove = ['challenge_blastConeOppositeOpponentCount', 'challenge_doubleAces', 'challenge_elderDragonKillsWithOpposingSoul', 
              'challenge_elderDragonMultikills', 'challenge_epicMonsterSteals', 'challenge_initialBuffCount', 
              'challenge_initialCrabCount', 'challenge_perfectGame', 'championName']
    
    for feature in remove:
        if feature in data.columns:
            data = data.drop(columns=[feature])
    
    with open(Data_dir / 'Updated_Data.csv', mode='w', encoding='utf-8') as file:
        data.to_csv(file, index=False)
        
def cleanup_8():
    '''
    1. remove outliers
    2. Save cleaned data to Updated_Data.csv
    '''
    
    data_path = Data_dir / 'Updated_Data.csv'
    data = pd.read_csv(data_path, low_memory=False)
    
    upper_limit = 3150
    lower_limit = 750
    
    data = data[(data['TOTAL_LP'] <= upper_limit) & (data['TOTAL_LP'] >= lower_limit)]
    
    with open(Data_dir / 'Updated_Data.csv', mode='w', encoding='utf-8') as file:
        data.to_csv(file, index=False)
        
def cleanup_9():
    '''
    1. Remove all columns with normalized variance less than 0.03
    2. Save cleaned data to Updated_Data.csv
    '''
    data_path = Data_dir / 'Updated_Data.csv'
    data = pd.read_csv(data_path, low_memory=False)
    
    threshold = 0.03
    for column in data.columns:
        if data[column].dtype == int or data[column].dtype == float:
            col = data[column]
            variance = col.var() / (col.max() - col.min())
            if variance < threshold:
                print(f"Dropping column: {column} with variance {variance}")
                data.drop(columns=[column], inplace=True)
    
    with open(Data_dir / 'Updated_Data.csv', mode='w', encoding='utf-8') as file:
        data.to_csv(file, index=False)
        
def cleanup_10():
    '''
    1. add /min data
    '''
    
    data_path = Data_dir / 'Updated_Data.csv'
    data = pd.read_csv(data_path, low_memory=False)
    
    minutes = data['gameDuration_s'] / 60.0
    alpha = 2
    minutes = minutes - alpha
    
    kills_per_min = data['kills'] / minutes
    deaths_per_min = data['deaths'] / minutes
    assists_per_min = data['assists'] / minutes
    gold_per_min = data['goldEarned'] / minutes
    cs_per_min = data['totalMinionsKilled'] / minutes
    damage_per_min = data['totalDamageDealtToChampions'] / minutes
    cs_first_10_per_min = data['challenge_laneMinionsFirst10Minutes'] / (10)
    
    data['kills_per_min'] = kills_per_min
    data['deaths_per_min'] = deaths_per_min
    data['assists_per_min'] = assists_per_min
    data['gold_per_min'] = gold_per_min
    data['cs_per_min'] = cs_per_min
    data['damage_per_min'] = damage_per_min
    data['cs_first_10_per_min'] = cs_first_10_per_min
    
    with open(Data_dir / 'Updated_Data.csv', mode='w', encoding='utf-8') as file:
        data.to_csv(file, index=False)
    
    

def cleanup_all():
    cleanup_1()
    cleanup_2()
    cleanup_3()
    cleanup_4()
    cleanup_5()
    cleanup_6()
    cleanup_7()
    cleanup_8()
    cleanup_9()
    cleanup_10()
    
if __name__ == "__main__":
    cleanup_all()