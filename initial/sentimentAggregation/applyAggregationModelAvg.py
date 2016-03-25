import json
import logging
import os
import pickle

import numpy as np


from utils.featuresStructure import featureStructureWorker
from sentimentAggregation.encodeAspects2Features import encodeAspects2features
from sentimentAggregation.encodeAspects2Features import encodeAspects1features


def applyAggregationModel(testReviews, featureAvgSent, model, busImportantFeatures, userImportantFeatures):
    logger = logging.getLogger('signature.aAM.applyAggregationModel')
    logger.info('starting applyAggregationModel from %d reviews'%len(testReviews))
    fsw = featureStructureWorker()
    
    for r, review in enumerate(testReviews):
        reviewFeatures = review['predSentiments']
        #features = encodeAspects2features(fsw, reviewFeatures)
        features = encodeAspects1features(fsw, reviewFeatures, featureAvgSent)
        #aggregation = model.predict(features)
        #print aggregation, review['stars']
        
        predictedFeatures = review['exPredFeatures']#[*,1]
        
        
        #Predicted Features, Predicted Sentiments by BUSINESS
        busID = review['business_id']
        if busID in busImportantFeatures:
            busSents =  busImportantFeatures[busID]['sentiment']
        else:
            busSents =  {}
        testData = {a:busSents[a][0] for a in busSents 
                    if a in predictedFeatures and predictedFeatures[a][1] == 1 and busSents[a][1] > 1}
        features = encodeAspects1features(fsw, testData, featureAvgSent)
        aggregationBUS = model.predict(features)
        
        review['rating_prediction'] = review.get('rating_prediction', {})
        review['rating_prediction']['aggregBUSavg'] = aggregationBUS
        
        
        if not r%10000:
            logger.debug('%d reviews processed'%r)
    
    return testReviews

def applyAM(path, model_num, limit = np.Inf):
    logger = logging.getLogger('signature.aAM')
    logger.info('starting applyAggregationModel')
    #get data
    r_file = path+'yelp_reviews_test_predictions.json'
    
    testReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        testReviews.append(json.loads(line.strip()))
    logger.info('Test Reviews loaded from %s'%r_file)
    
    #load model
    model = pickle.load(open(path+'aggregationAvg_%d.model'%model_num,'r'))
    
    infile = open(path+'trainSentimentAverages.json','r')
    featureAvgSent = json.loads(infile.readline())
    infile.close()
    
    
    #get data
    b_file = path+'businessFeaturesAggregation_stat.json'
    u_file = path+'userFeaturesAggregation_stat.json'
    
    busImportantFeatures = json.loads(open(b_file,'r').readline())
    logger.info('Important BUSINESS Features loaded')
    userImportantFeatures = json.loads(open(u_file,'r').readline())
    logger.info('Important USER Features loaded')
    
    #run function
    reviewsPrediction = applyAggregationModel(testReviews, featureAvgSent, model,
                                              busImportantFeatures, userImportantFeatures)
    
    #save result
    outfile = open(path+'yelp_reviews_test_predictions.json','w')
    for review in reviewsPrediction:
        outfile.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
    outfile.close()
    