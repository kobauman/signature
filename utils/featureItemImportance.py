import json
import logging
import sys
import math
import copy
sys.path.append('../')

from utils.featuresStructure import featureStructureWorker

'''
get dictionary of Items (users or restaurants) with lists of reviews with assigned features
return 
    1) set of highly important features
    2) set of frequent features
    3) set of TF-IDF important features
'''


def featureImportance(review_dict):
    logger = logging.getLogger('signature.fI')
    logger.info('starting featureImportance')
    fsw = featureStructureWorker()
    N = len(review_dict)
    featuresDF = dict()
    #itemTemplate = {'tfidfDict':{},'featureFreq':{},'reviewsNumber':0}
    itemFeatures = dict()
    for item in review_dict:
        itemFeatures[item] = {'tfidfDict':{},'featureFreq':{},'reviewsNumber':0, 'maxFreq':0}
        itemFeatures[item]['reviewsNumber'] = len(review_dict[item])
        for review in review_dict[item]:
            reviewFeatures = fsw.getReviewFeatures(review)
            for feature in reviewFeatures:
                itemFeatures[item]['featureFreq'][feature] = itemFeatures[item]['featureFreq'].get(feature,0)
                itemFeatures[item]['featureFreq'][feature] += 1
        for feature in itemFeatures[item]['featureFreq']:
            if itemFeatures[item]['featureFreq'][feature] > itemFeatures[item]['maxFreq']:
                itemFeatures[item]['maxFreq'] = itemFeatures[item]['featureFreq'][feature]
            featuresDF[feature] = featuresDF.get(feature, 0)
            featuresDF[feature] += 1
        
    #prepare IDF
    for feature in featuresDF:
        featuresDF[feature] = math.log(float(N)/featuresDF[feature])
    
    for item in itemFeatures:
        for feature in itemFeatures[item]['featureFreq']:
            tf = float(itemFeatures[item]['featureFreq'][feature])/itemFeatures[item]['maxFreq']
            #print feature, tf
            idf = featuresDF[feature]
            itemFeatures[item]['tfidfDict'][feature] = round(tf*idf, 3)
            Ni = len(review_dict[item])
            t = round(100.*itemFeatures[item]['featureFreq'][feature]/Ni,2)
            itemFeatures[item]['featureFreq'][feature] = t
            
        itemFeatures[item]['tfidfList'] = [[itemFeatures[item]['tfidfDict'][feature],feature] 
                                           for feature in itemFeatures[item]['tfidfDict']]
        
        itemFeatures[item]['tfidfList'].sort(reverse = True)
        
        
        itemFeatures[item]['featureFreqList'] = [[itemFeatures[item]['featureFreq'][feature],feature] 
                                           for feature in itemFeatures[item]['featureFreq']]
        
        itemFeatures[item]['featureFreqList'].sort(reverse = True)
    
    return copy.deepcopy(itemFeatures)