import json
import logging
import os
import pickle

import numpy as np


from utils.featuresStructure import featureStructureWorker
from sentimentAggregation.learnAggregationModel import encodeAspects2features


def applyAggregationModel(testReviews, model):
    logger = logging.getLogger('signature.aAM.applyAggregationModel')
    logger.info('starting applyAggregationModel from %d reviews'%len(testReviews))
    fsw = featureStructureWorker()
    
    for r, review in enumerate(testReviews):
        reviewFeatures = review['predSentiments']
        features = encodeAspects2features(fsw, reviewFeatures)
        aggregation = model.predict(features)
        #print aggregation, review['stars']
        review['rating_prediction'] = aggregation
        
        if not r%10000:
            logger.debug('%d reviews processed'%r)
    
    return testReviews

def applyAM(path, model_num, limit = np.Inf):
    logger = logging.getLogger('signature.aAM')
    logger.info('starting applyAggregationModel')
    #get data
    r_file = path+'/yelp_reviews_features_test_pF_sent.json'
    
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
    
    #run function
    reviewsPrediction = applyAggregationModel(testReviews, model)
    
    #save result
    outfile = open(path+'/yelp_reviews_features_test_pF_sent_agg.json','w')
    for review in reviewsPrediction:
        outfile.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
    outfile.close()
    