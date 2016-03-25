import json
import logging
import os


import numpy as np
import graphlab

from utils.featuresStructure import featureStructureWorker



def applySentimentMF(testReviews,  modelDict_ex, featureThres_ex, modelDict, featureThres):
    logger = logging.getLogger('signature.aSMF.applySentimentMF')
    logger.info('starting applySentimentMatrixFactorization from %d reviews'%len(testReviews))
    fsw = featureStructureWorker()
    
    
    aspect_data = dict()
    reviewDict_ex = dict()
    reviewDict = dict()
    
    for r, review in enumerate(testReviews):
        busID = review['business_id']
        userID = review['user_id']
        reviewID = review['review_id']
        for aspect in fsw.featureIdicator:
            if not fsw.featureIdicator.get(aspect, None):
                continue
            
            if aspect in aspect_data:
                aspect_data[aspect]['id'].append(reviewID)
                aspect_data[aspect]['user'].append(userID)
                aspect_data[aspect]['item'].append(busID)
            else:
                aspect_data[aspect] = {'id':[reviewID],'user':[userID],'item':[busID]}
        if not r%5000:
            logger.debug('%d reviews processed'%r)   
   
   
    for f, aspect in enumerate(aspect_data):
        logger.info('Prosessing  (%d) %s'%(f,aspect))
        if aspect not in modelDict_ex or aspect not in modelDict:
            continue
        testData = graphlab.SFrame(aspect_data[aspect])
#        print('test prepared')
        prediction_ex = modelDict_ex[aspect].predict(testData)
        prediction = modelDict[aspect].predict(testData)
#        print('sentiment predicted')
        testData['prediction_ex'] = prediction_ex
        testData['prediction'] = prediction
        
        #existence
        testData_prediction_ex = list(testData['prediction_ex'])
        for i,prediction_ex in enumerate(testData_prediction_ex):
            reviewID = aspect_data[aspect]['id'][i]
            reviewDict_ex[reviewID] = reviewDict_ex.get(reviewID,{})
            
            ex_pred_adjust = (prediction_ex*0.5/featureThres_ex[aspect])
            if ex_pred_adjust < 0:
                ex_pred_adjust = 0
            if ex_pred_adjust > 1:
                ex_pred_adjust = 1
            reviewDict_ex[reviewID][aspect] = ex_pred_adjust
        
        #sentiment
        testData_prediction = list(testData['prediction'])
        for i,sent_prediction in enumerate(testData_prediction):
            reviewID = aspect_data[aspect]['id'][i]
            reviewDict[reviewID] = reviewDict.get(reviewID,{})
            
            sent_pred_adjust = (sent_prediction*0.5/featureThres[aspect])
            if sent_pred_adjust < 0:
                sent_pred_adjust = 0
            if sent_pred_adjust > 1:
                sent_pred_adjust = 1
            reviewDict[reviewID][aspect] = sent_pred_adjust
            
        if not f%1:
            logger.debug('%d features sentiments predicted'%f)
    
    return reviewDict_ex, reviewDict

def applySMF(path, limit = np.Inf):
    logger = logging.getLogger('signature.applySentimentMF')
    logger.info('starting applySentimentMF')
    #get data
    r_file = path+'specific_reviews_test.json'
    
    testReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%5000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        testReviews.append(json.loads(line.strip()))
    logger.info('Test Reviews loaded from %s'%r_file)
    
    
    #load model existence
    modelDict_ex = dict()
    featureThres_ex = dict()
    fsw = featureStructureWorker()
    for feature in fsw.featureIdicator:
        if not fsw.featureIdicator[feature]:
            continue
        try:
            modelPath = path + 'sentimentModels/%s_sentiment_ex.model'%feature
            modelDict_ex[feature] = graphlab.load_model(modelPath)
            
            #load average
            thres_path = path+'sentimentModels/%s_sentiment_ex.threshold'%feature
            infile = open(thres_path,'r')
            featureThres_ex[feature] = float(infile.readline().strip())
            infile.close()
        except:
            logger.error('There is no model for feature: %s'%feature)
            continue
        
    logger.info('Existence Models loaded')
    
    
    #load model
    modelDict = dict()
    featureThres = dict()
    fsw = featureStructureWorker()
    for feature in fsw.featureIdicator:
        if not fsw.featureIdicator[feature]:
            continue
        try:
            modelPath = path + 'sentimentModels/%s_sentiment.model'%feature
            print modelPath
            modelDict[feature] = graphlab.load_model(modelPath)
            
            #load average
            thres_path = path+'sentimentModels/%s_sentiment.threshold'%feature
            infile = open(thres_path,'r')
            featureThres[feature] = float(infile.readline().strip())
            infile.close()
        except:
            logger.error('There is no model for feature: %s'%feature)
            continue
        
    logger.info('Sentiment Models loaded')
        
    #run function
    results_ex, results = applySentimentMF(testReviews, modelDict_ex, featureThres_ex, modelDict, featureThres)
    
    #save result
    json.dump(results_ex,open(path+'reviews_test_exMFpred.json','w'))
    json.dump(results,open(path+'reviews_test_MFpred.json','w'))