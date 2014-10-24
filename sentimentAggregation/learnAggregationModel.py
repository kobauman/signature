import json
import logging
import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.metrics import precision_recall_curve

from utils.featuresStructure import featureStructureWorker
from sentimentAggregation.encodeAspects2Features import encodeAspects2features
from sentimentAggregation.encodeAspects2Features import encodeAspects1features



def learnAggregationModelsCV(trainReviews, featureAvgSent, path):
    logger = logging.getLogger('signature.lAMCV.learnAggregationModelsCV')
    logger.info('starting learnAggregationModel from %d reviews'%len(trainReviews))
    fsw = featureStructureWorker()
    
    learnData = list()
    learnLabels = list()
    for j, review in enumerate(trainReviews):
        reviewFeatures = fsw.getReviewFeaturesSentiment(review['features'])
        rating = review['stars']
        #features = encodeAspects2features(fsw, reviewFeatures)
        features = encodeAspects1features(fsw, reviewFeatures, featureAvgSent)
        learnData.append(features)
        learnLabels.append(rating)
    
    learnData = np.array(learnData)
    learnLabels = np.array(learnLabels)
    
    bestRes = 0.0
    bestReg = 0.0
    for reg in [0.01,0.05,0.1,0.2,0.5,1.0,5.0,10,15,50,100,200,500]:
        kf = cross_validation.KFold(len(learnLabels), n_folds=10)
        results = list()
        for train_index, test_index in kf:
            X_train, X_test = learnData[train_index], learnData[test_index]
            y_train, y_test = learnLabels[train_index], learnLabels[test_index]
            clf = linear_model.Ridge(alpha = reg)
            clf.fit (X_train, y_train)
            results.append(clf.score(X_test, y_test))
        if np.average(results) > bestRes:
            bestRes = np.average(results)
            bestReg = reg
        #print reg, np.average(results)
    logger.info('Best score %f with regularization = %.2f'%(bestRes, bestReg))
    
    clf = linear_model.Ridge(alpha = bestReg)
    clf.fit(learnData, learnLabels)
    
    return clf


def learnAggreationModel(path, limit = np.Inf):
    logger = logging.getLogger('signature.lAMCV')
    logger.info('starting learnAggreationModel')
    #get data
    r_file = path+'/yelp_reviews_features_train.json'
    
    trainReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        trainReviews.append(json.loads(line.strip()))
    
    infile = open(path+'trainSentimentAverages.json','r')
    featureAvgSent = json.loads(infile.readline())
    infile.close()
    
    #run function
    model = learnAggregationModelsCV(trainReviews, featureAvgSent, path)
    
    #save model
    pickle.dump(model,open(path+'aggregation_%d.model'%counter,'wb'))
    
    return counter
    