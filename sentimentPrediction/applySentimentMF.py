import json
import logging
import pickle

import numpy as np
import graphlab as gl

from utils.featuresStructure import featureStructureWorker



        


def applySentimentMF(testReviews, modelDict):
    logger = logging.getLogger('signature.aSMF.applySentimentMF')
    logger.info('starting applySentimentMatrixFactorization from %d reviews'%len(testReviews))
    fsw = featureStructureWorker()
    
    
    feature_data = dict()
    reviewDict = dict()
    
    for r, review in enumerate(testReviews):
        review['predSentiments'] = dict()
        busID = review['business_id']
        userID = review['user_id']
        sentiments = fsw.getReviewFeaturesSentiment(review['features'])
        #print sentiments
        ID = busID+'###'+userID
        reviewDict[ID] = review
        for feature in review['exPredFeatures']:
            if not fsw.featureIdicator.get(feature, None):
                continue
            sentiment = np.average(sentiments.get(feature,[0.0]))
            if feature in feature_data:
                feature_data[feature]['id'].append(ID)
                feature_data[feature]['user'].append(userID)
                feature_data[feature]['item'].append(busID)
                feature_data[feature]['rating'].append(sentiment)
            else:
                feature_data[feature] = {'id':[ID],'user':[userID],'item':[busID],'rating':[sentiment]}
        if not r%1000:
            logger.debug('%d reviews processed'%r)   
             
    for f, feature in enumerate(feature_data):
        #print feature, feature_data[feature]
        testData = gl.SFrame(feature_data[feature])
        prediction = modelDict[feature].score(testData)
        testData['prediction'] = prediction
        
        for i,ID in enumerate(testData['id']):
            reviewDict[ID]['predSentiments'][feature] = testData['prediction'][i]
        
        if not f%1:
            logger.debug('%d features sentiments predicted'%f)
    
    return [reviewDict[i] for i in reviewDict]


def applySMF(path, limit):
    logger = logging.getLogger('signature.aSMF')
    logger.info('starting applySentimentMF')
    #get data
    r_file = path+'/yelp_reviews_features_test_pF.json'
    
    testReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        testReviews.append(json.loads(line.strip()))
    logger.info('Test Reviews loaded from %s'%r_file)
    
    #load model
    modelDict = dict()
    fsw = featureStructureWorker()
    for feature in fsw.featureIdicator:
        if not fsw.featureIdicator[feature]:
            continue
        try:
            modelPath = path + '/sentimentModels/%s_sentiment.model'%feature
            modelDict[feature] = gl.load_model(modelPath)
        except:
            logger.error('There is no model for feature: %s'%feature)
            continue
    logger.info('Models loaded')
    
    #run function
    reviewsPrediction = applySentimentMF(testReviews, modelDict)
    
    #save result
    outfile = open(path+'/yelp_reviews_features_test_pF_sent.json','w')
    for review in reviewsPrediction:
        outfile.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
    outfile.close()