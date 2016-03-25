import json
import logging
import pickle
import random
import os
import numpy as np

from utils.featuresStructure import featureStructureWorker
from featureWorkers.getFeatures import calculateFeatures


def predictAll(path, modelfile):
    logger = logging.getLogger('signature.pairCompare')
    logger.info('starting pairCompare')
    #get data
    b_file = path+'/businessProfile.json'
    u_file = path+'/userProfile.json'
    r_file = path+'/specific_reviews_test.json'
    
    fsw = featureStructureWorker()
    
    #load model
    modelDict = pickle.load(open(modelfile,'rb'))
    logger.info('Model loaded from %s'%modelfile)
    
    
    busImportantFeatures = json.loads(open(b_file,'r').readline())
    logger.info('Important BUSINESS Features loaded')
    userImportantFeatures = json.loads(open(u_file,'r').readline())
    logger.info('Important USER Features loaded')
    testReviewsByUser = dict()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        
        review = json.loads(line.strip())
        userID = review['user_id']
        
        for aspect in modelDict:
            if not fsw.featureIdicator[aspect]:
                continue
            
            featureSet = calculateFeatures(logger, review, aspect, busImportantFeatures, userImportantFeatures)
            if not featureSet:
                continue
            
            review['pairComp'] = review.get('pairComp', {})
            predProb = modelDict[aspect][1].predict_proba(np.array([featureSet]))[0][1]
            
            if predProb > 0.5:
                predSent = modelDict[aspect][3].predict_proba(np.array([featureSet]))[0][1]
                
                review['pairComp'][aspect] = predSent
            
            #print(review['pairComp'])
            
        testReviewsByUser[userID] = testReviewsByUser.get(userID, [])
        testReviewsByUser[userID].append(review)
    
    logger.info('Reviews loaded')
    
    
    
    
    #save result
    outfile = open(path+'test_predictions.json','w')
    for user in testReviewsByUser:
        outfile.write(json.dumps(testReviewsByUser[user])+'\n')
    outfile.close()