import sys, os
from lib2to3.pgen2.token import MINUS, PLUS
sys.path.append('../')
sys.path.append('../../')


import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from utils.getAspectList import getSetOfAspects

# appName = 'beautySpa'
# appName = 'hotel'
appName = 'restaurant'
 
path = '../../../data/'

def busAspectsCollocation(asp1,asp2,bus_df):
    l = len(bus_df)
    df1 = bus_df[bus_df[asp1]==1]
    df2 = bus_df[bus_df[asp2]==1]
    if l < 10 or len(df1)/l < 0.1 or len(df2)/l < 0.1:
        return None
    
    df12 = df1[df1[asp2]==1]
    
    return len(df12)/l

# @profile
def regression(aspects, dataset):
    asps = list(set(dataset.columns).intersection(aspects))
    asps.sort()
    
    aspsMP = list()
    for asp in asps:
        minus = asp+'_minus'
        dataset[minus] = dataset.apply(lambda x: 1 if x[asp] and x[asp+'sent'] == -1 else 0, axis=1)    
        aspsMP.append(minus)
         
        plus = asp+'_plus'
        dataset[plus] = dataset.apply(lambda x: 1 if x[asp] and x[asp+'sent'] == 1 else 0, axis=1)    
        aspsMP.append(plus)
         
#         overall = 'a_'+asp
#         dataset[overall] = dataset.apply(lambda x: x[asp]*x[asp+'sent'], axis=1)    
#         aspsMP.append(overall)

        neutral = asp+'_neutral'
        dataset[neutral] = dataset.apply(lambda x: 1 if x[asp] and x[asp+'sent'] == 0 else 0, axis=1)    
        aspsMP.append(neutral)
        
#         aspsMP.append(asp+'sent')
        
        
#     MINUS
#     PLUS
    
    aspsMP.sort()
    dataset['intercept'] = np.ones(len(dataset))
    aspsMP = ['intercept'] + aspsMP
#     print(len(aspects),len(asps))
    model = OLS(dataset['stars'], dataset[aspsMP]).fit()
#     model.summary
#     print(model.params)
#     print(model.pvalues)
    
    return model.summary()


# 1) get list of features
app_aspects = getSetOfAspects(appName)
#print(app_aspects)
print('Aspects loaded.')

# 2) load data
reviews_file = os.path.join(path,'BING_DATASET', appName+'_data.csv')
df = pd.DataFrame.from_csv(reviews_file).head(100000).copy()
for asp in app_aspects:
    if asp in df.columns:
        df[asp] = df[asp].fillna(0.0)
        df[asp+'sent'] = df[asp+'sent'].fillna(-10.0)
        
# print(df.columns)
print('Dataset loaded.')


# 3) fit regression model

# !!! get random sample
ac = regression(app_aspects, df.head(100000000))
output = os.path.join(path,'ASPECT_RELATIONS', appName+'_regression.txt')
with open(output,'w') as o:
    o.write(str(ac))
print('Regression model fitted.')
