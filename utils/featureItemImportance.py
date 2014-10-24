import json
import logging
import sys
import math
import copy
import numpy as np
sys.path.append('../')

from utils.featuresStructure import featureStructureWorker

'''
get dictionary of Items (users or restaurants) with lists of reviews with assigned features
return 
    1) set of highly important features
    2) set of frequent features
    3) set of TF-IDF important features
'''


def featureImportance(review_dict, ignore_neutral = True):
    logger = logging.getLogger('signature.IFI.fI')
    logger.info('starting featureImportance')
    fsw = featureStructureWorker()
    N = len(review_dict)
    featuresDF = dict() # dictionary for counting Document Frequency
    itemFeatures = dict() # main dictionary with statistics (item:stat)
    for it, item in enumerate(review_dict):
        itemFeatures[item] = {'tfidfDict':{},'featureFreq':{},'sentiment':{},'reviewsNumber':0, 'maxFreq':0, 'textFeatures':[]}
        itemFeatures[item]['reviewsNumber'] = len(review_dict[item])
        for r, review in enumerate(review_dict[item]):
            reviewFeatures = fsw.getReviewFeaturesExistence(review['features'])
            for feature in reviewFeatures:
                #work with frequency
                itemFeatures[item]['featureFreq'][feature] = itemFeatures[item]['featureFreq'].get(feature,0)
                itemFeatures[item]['featureFreq'][feature] += 1
                #work with sentiment
                itemFeatures[item]['sentiment'][feature] = itemFeatures[item]['sentiment'].get(feature,[])
                if len(reviewFeatures[feature]):
                    if ignore_neutral:
                        arr = [x for x in reviewFeatures[feature] if x]
                        if len(arr):
                            itemFeatures[item]['sentiment'][feature].append(np.average(arr))
                        else:
                            itemFeatures[item]['sentiment'][feature].append(0.0)
                    else:
                        itemFeatures[item]['sentiment'][feature].append(np.average(reviewFeatures[feature]))
                else:
                    itemFeatures[item]['sentiment'][feature].append(0.0)
            
            
            #print review.keys()
            if not len(itemFeatures[item]['textFeatures']):
                for tf in review['textFeatures']:
                    itemFeatures[item]['textFeatures'].append(tf)
            else:
                for i,tf in enumerate(review['textFeatures']):
                    itemFeatures[item]['textFeatures'][i] += tf
            
#            if not r%10:
#                logger.debug('%d reviews'%r)
            
                    
        for feature in itemFeatures[item]['featureFreq']:
            #work with frequency
            if itemFeatures[item]['featureFreq'][feature] > itemFeatures[item]['maxFreq']:
                itemFeatures[item]['maxFreq'] = itemFeatures[item]['featureFreq'][feature]
            #work with sentiment
            itemFeatures[item]['sentiment'][feature] = [round(np.average(itemFeatures[item]['sentiment'][feature]),3),
                                     len(itemFeatures[item]['sentiment'][feature])]
            #work with 'Document' Frequency (DF)
            featuresDF[feature] = featuresDF.get(feature, 0)
            featuresDF[feature] += 1
        
        for tf in range(len(itemFeatures[item]['textFeatures'])):
            itemFeatures[item]['textFeatures'][tf] /= itemFeatures[item]['reviewsNumber']
        
        if not it%1000:
            logger.debug('%d items'%it)
            
    #prepare IDF
    for feature in featuresDF:
        featuresDF[feature] = math.log(float(N)/featuresDF[feature])
    
    logger.debug('IDF prepared for %d items'%it)
    
    for it, item in enumerate(itemFeatures):
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
        
        if not it%1000:
            logger.debug('%d items completed'%it)
    return copy.deepcopy(itemFeatures)