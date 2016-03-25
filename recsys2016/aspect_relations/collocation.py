import sys, os
sys.path.append('../')
sys.path.append('../../')


import numpy as np
import pandas as pd

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
@profile
def aspectCollocation(aspects, dataset):
    aspect_relations = list()
    
    aspect_df = dict()
    bus_df = dict()
    
    for a1 in range(len(aspects)):
        aspect1 = aspects[a1]
        if aspect1 not in dataset.columns:
            continue
        print(a1,aspect1)
        aspect_df[aspect1] = aspect_df.get(aspect1,dataset[dataset[aspect1]==1])
        df1 = aspect_df[aspect1]
        
        for a2 in range(a1+1,len(aspects)):
            aspect2 = aspects[a2]
            if aspect2 not in dataset.columns:
                continue
            print('\t',a2,aspect2)
            
#             aspect_df[aspect2] = aspect_df.get(aspect2,dataset[dataset[aspect2]==1])
#             df2 = aspect_df[aspect2]
                        
            df12 = df1[df1[aspect2]==1]
            common_bus_set = df12['business_id'].unique()
            
            b_collocations_list = list()
            b_collocations1_list = list()
            b_collocations2_list = list()
            
            frequency_1 = list()
            frequency_2 = list()
            t = None
            for bus in common_bus_set:
                bus_df[bus] = bus_df.get(bus, dataset[dataset['business_id']==bus])
                
                l = len(bus_df[bus])
                if l < 50:
                    continue
                
                bdf1 = bus_df[bus][bus_df[bus][aspect1]==1]
                ldf1 = len(bdf1)
                
                if ldf1/l < 0.1:
                    continue
                
                bdf2 = bus_df[bus][bus_df[bus][aspect2]==1]
                ldf2 = len(bdf2)
                
                if ldf2/l < 0.1:
                    continue
                
                bdf12 = bdf1[bdf1[aspect2]==1]
                ldf12 = len(bdf12)
                
                if l > 50 and ldf1/l > 0.1 and ldf2/l > 0.1:
#                     print(aspect1,aspect2,ldf1,ldf2,ldf12,l)
                    b_collocations_list.append(ldf12/l)
                    b_collocations1_list.append(ldf12/ldf1)
                    frequency_1.append(ldf1/l)
                    b_collocations2_list.append(ldf12/ldf2)
                    frequency_2.append(ldf2/l)
                    t = (ldf1,ldf2,ldf12,l)
                    
                    
            if len(b_collocations_list) == 0:
                continue
            if len(b_collocations_list) == 1:
                print(t,frequency_1,frequency_2,b_collocations1_list,b_collocations2_list)
            aspect_relations.append({'aspect1':aspect1,'aspect2':aspect2,
                                     'bus_set':len(common_bus_set),
                                     'b_coll_list':len(b_collocations_list),
                                     'collocation':round(np.average(b_collocations_list),4),
                                     'coll_2_in_1':round(np.average(b_collocations1_list),4),
                                     'frquency_1':round(np.average(frequency_1),4),
                                     'coll_1_in_2':round(np.average(b_collocations2_list),4),
                                     'frquency_2':round(np.average(frequency_2),4)
                                     })
            if a2 > 5:
                break
        if a1 > 5:
            break
    return pd.DataFrame(aspect_relations)


# 1) get list of features
app_aspects = getSetOfAspects(appName)
#print(app_aspects)


# 2) load data
reviews_file = os.path.join(path,'BING_DATASET', appName+'_data.csv')
df = pd.DataFrame.from_csv(reviews_file).head(30000)

# 3) 
ac = aspectCollocation(app_aspects, df)
output = os.path.join(path,'ASPECT_RELATIONS', appName+'_cololocations.csv')
ac.to_csv(output,index=False)
