import json
import logging
import os
import pickle

import numpy as np


from utils.featuresStructure import featureStructureWorker
from sentimentAggregation.encodeAspects2Features import encodeAspects2features
from sentimentAggregation.encodeAspects2Features import encodeAspects1features


def applyAggregationModel(testReviews, featureAvgSent, model):
    logger = logging.getLogger('signature.aAM.applyAggregationModel')
    logger.info('starting applyAggregationModel from %d reviews'%len(testReviews))
    fsw = featureStructureWorker()
    
    for r, review in enumerate(testReviews):
        reviewFeatures = review['predSentiments']
        #features = encodeAspects2features(fsw, reviewFeatures)
        features = encodeAspects1features(fsw, reviewFeatures, featureAvgSent)
        #aggregation = model.predict(features)
        #print aggregation, review['stars']
        
        realFeatures = fsw.getReviewFeaturesSentiment(review['features'])
        predictedFeatures = review['exPredFeatures']#[*,1]
        predictedSentiments = review['predSentiments']
        
        #best aggregtion
        #REAL Features, REAL Sentiments
        testData = {a:np.average(realFeatures[a]) for a in realFeatures}
        features = encodeAspects1features(fsw, testData, featureAvgSent)
        aggregationRR = model.predict(features)
        
        #Predicted Features, REAL Sentiments
        testData = {a:np.average(realFeatures[a]) for a in realFeatures if predictedFeatures[a][1] == 1}
        features = encodeAspects1features(fsw, testData, featureAvgSent)
        aggregationPR = model.predict(features)
        
        #REAL Features, Predicted Sentiments
        testData = {a:predictedSentiments[a] for a in predictedSentiments if a in realFeatures}
        features = encodeAspects1features(fsw, testData, featureAvgSent)
        aggregationRP = model.predict(features)
        
        #Predicted Features, Predicted Sentiments
        testData = {a:predictedSentiments[a] for a in predictedSentiments 
                    if a in predictedFeatures and predictedFeatures[a][1] == 1}
        features = encodeAspects1features(fsw, testData, featureAvgSent)
        aggregationPP = model.predict(features)
        
        
        review['rating_prediction'] = [aggregationRR, aggregationPR, aggregationRP, aggregationPP]
        
        if not r%10000:
            logger.debug('%d reviews processed'%r)
    
    return testReviews

def applyAM(path, model_num, limit = np.Inf):
    logger = logging.getLogger('signature.aAM')
    logger.info('starting applyAggregationModel')
    #get data
    r_file = path+'yelp_reviews_features_test_pF_sent.json'
    
    testReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        testReviews.append(json.loads(line.strip()))
    logger.info('Test Reviews loaded from %s'%r_file)
    
    #load model
    model = pickle.load(open(path+'aggregation_%d.model'%model_num,'r'))
    
    infile = open(path+'trainSentimentAverages.json','r')
    featureAvgSent = json.loads(infile.readline())
    infile.close()
    
    #run function
    reviewsPrediction = applyAggregationModel(testReviews, featureAvgSent, model)
    
    #save result
    outfile = open(path+'yelp_reviews_features_test_pF_sent_agg.json','w')
    for review in reviewsPrediction:
        outfile.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
    outfile.close()
    