import json
import logging
import pickle
import random
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

import sys
sys.path.append('../')


from utils.featuresStructure import featureStructureWorker
from featureWorkers.getFeatures import getFeatures
from featureWorkers.getFeatures import calculateFeatures


def sig_dif(array1,array2):
    try:
        arr1 = np.array(array1)
        arr2 = np.array(array2)
        return round(stats.ttest_ind(arr1,arr2)[1],3)
    except:
        return None

def getWeight(aspect, profile):
    return random.random()
    if aspect in profile['critical']:
        pos = profile['critical'][aspect]['+']
        neg = profile['critical'][aspect]['-']
            
        if int(sig_dif(pos,neg) < 0.0501):
            return np.average(pos) - np.average(neg)
        
    return 0.0




def reviewCompare(review1,review2, aspectWeight = {}):
    aspectUnion = set(review1.keys()).union(set(review2.keys()))
    aspectIntersection = set(review1.keys()).intersection(set(review2.keys()))
#    print(aspectUnion)
    union = 0.0
    inter = 0.0
    inter_num = 0
    dom1 = 0.0
    dom2 = 0.0
    for aspect in aspectUnion:
        union += aspectWeight.get(aspect, 1)
        if aspect in aspectIntersection:
            inter += aspectWeight.get(aspect, 1)
            inter_num += 1
            
            if review1[aspect] > review2[aspect]:
                dom1 += aspectWeight.get(aspect, 1)
            if review2[aspect] > review1[aspect]:
                dom2 += aspectWeight.get(aspect, 1)
    
    if union == 0:
        return -1
    jaccard = inter / union
#    print(jaccard)
    if jaccard < 0.3 or inter_num < 3:
        return -1
#    if inter_num < 3:
#        return -1
    
    dominance1 = dom1 / inter
    dominance2 = dom2 / inter
#    print(dominance1,dominance2)

    #random
    if dominance1 > 0.8 or dominance2 > 0.8:
        return  random.randint(1,2)

    if dominance1 > 0.8:
        return 1
    elif dominance2 > 0.8:
        return 2
    else:
        return 0
    
    
    
def pairCompare(path, modelfile):
    logger = logging.getLogger('signature.pairCompare')
    logger.info('starting pairCompare')
    
#    b_file = path+'/businessProfile.json'
#    u_file = path+'/userProfile.json'
#    busImportantFeatures = json.loads(open(b_file,'r').readline())
#    logger.info('Important BUSINESS Features loaded')
#    userImportantFeatures = json.loads(open(u_file,'r').readline())
#    logger.info('Important USER Features loaded')
#    
#    #load model
#    modelDict = pickle.load(open(modelfile,'rb'))
#    logger.info('Model loaded from %s'%modelfile)
    
    
    
    nonComparible = 0
    comparible = 0
    correct = 0
    equal_stars = 0
    equal = 0
    
    compareList = [[],[],[]]
    
    for counter, user_line in enumerate(open(path+'test_predictions.json','r')):
        reviews = json.loads(user_line)
        
        if len(reviews) < 2:
            continue
#        userID = reviews[0]['user_id']
#        if userID not in userImportantFeatures:
#            continue
        
#        userProfile = userImportantFeatures[userID]
#        aspectWeight = {}
#        for aspect in modelDict:
#            aspectWeight[aspect] = getWeight(aspect, userProfile)
#        print(userID, aspectWeight)
        
        for i in range(len(reviews)):
#            busIDi = reviews[i]['business_id']
#            if busIDi not in busImportantFeatures:
#                continue
#            busProfile_i = busImportantFeatures[busIDi]
#            aspectWeight_i = {}
#            for aspect in modelDict:
#                aspectWeight_i[aspect] = getWeight(aspect, busProfile_i)
#                
            for j in range(i+1,len(reviews)):
                if 'pairComp' not in reviews[i] or 'pairComp' not in reviews[j]:
                    nonComparible += 1
                    continue
                if len(reviews[i]['pairComp']) == 0 or len(reviews[j]['pairComp']) == 0:
                    nonComparible += 1
                    continue
                
#                busIDj = reviews[j]['business_id']
#                if busIDj not in busImportantFeatures:
#                    continue
#                busProfile_j = busImportantFeatures[busIDj]
#                aspectWeight_j = {}
#                for aspect in modelDict:
#                    aspectWeight_j[aspect] = getWeight(aspect, busProfile_j)
#                    
#                
#                aspectWeight_final = {aspect: np.average([aspectWeight_j[aspect]+aspectWeight_j[aspect]]) for aspect in aspectWeight_j}
##                print(aspectWeight_final)
                
                #print(reviews[i]['pairComp'],reviews[j]['pairComp'])
                compareCoef = reviewCompare(reviews[i]['pairComp'],reviews[j]['pairComp'])#, aspectWeight_final)
                if compareCoef == 1:
                    compareList[0].append(reviews[i]['stars'])
                    compareList[1].append(reviews[j]['stars'])
                    compareList[2].append(reviews[i]['stars']-reviews[j]['stars'])
                    comparible += 1
                    if reviews[i]['stars']-reviews[j]['stars'] >= 0:
                        correct += 1
                    if reviews[j]['stars'] == reviews[i]['stars']:
                        equal_stars += 1
                elif compareCoef == 2:
                    compareList[0].append(reviews[j]['stars'])
                    compareList[1].append(reviews[i]['stars'])
                    compareList[2].append(reviews[j]['stars']-reviews[i]['stars'])
                    comparible += 1
                    if reviews[j]['stars']-reviews[i]['stars'] >= 0:
                        correct += 1
                    if reviews[j]['stars'] == reviews[i]['stars']:
                        equal_stars += 1
                elif compareCoef == 0:
                    equal += 1
                elif compareCoef == -1:
                    nonComparible += 1
                    
    
    print([np.average(x) for x in compareList])
    print(len(compareList[0]))
    
    #compareList significant difference
    difference = 'Average winner\'s stars = %.3f; \nAverage looser\'s stars = %.3f\nAverage difference = %.3f\nSignificance = %.3f'%(np.average(compareList[0]),
                                                                                                                                     np.average(compareList[1]),
                                                                                                                                     np.average(compareList[2]),
                                                                                                                                     sig_dif(compareList[0],compareList[1]))
    difference += '\nCorrect = %d\nAccuracy = %.3f'%(correct,correct/comparible)
    difference += '\nEqual stars = %d\nAccuracy_2 = %.3f'%(equal_stars,equal_stars/comparible)
    
    #statistics
    stat = '\nNonComparible = %d\nComparible = %d\nEqual = %d\n'%(nonComparible,comparible,equal)
    
    print(difference)
    print(stat)
    try:
        os.stat(path+'results/')
    except:
        os.mkdir(path+'results/')
        
    
    #save feature Quality
    outfile = open(path+'/results/pairCompare.txt','w')
    outfile.write(difference)
    outfile.write(stat)
    outfile.close()
    
    
    #compareList histogram
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 9))
    
    ax0, ax1 = axes.flat
    
    n_bins = 5
    
    ax0.hist(compareList[:2], n_bins, normed=0,alpha=0.5, histtype='bar', label=['Winners','Loosers'], range=(0.5, 5.5))
    #ax0.hist(compareList[1], n_bins, normed=1,alpha=0.5, histtype='bar', color='r', label='Loosers', range=(0.5, 5.5))
    ax0.legend(prop={'size': 10})
    ax0.set_title('Star ratings')
    
    n_bins = 9
    ax1.hist(compareList[2], n_bins, normed=1, histtype='bar', range=(-4.5, 4.5))
    ax1.set_title('Difference')
    
    try:
        os.stat(path+'/testPictures/')
    except:
        os.mkdir(path+'/testPictures/')
    plt.savefig(path+'/testPictures/pairCompare.png')
