import json
import logging

import numpy as np
import graphlab as gl

from utils.featuresStructure import featureStructureWorker


def getBestReg(logger, df):
    for i in range(3):
        (train_set, test_set) = df.random_split(0.8)
        regularization_vals = [10,100]
        feats = [2,5,7,10]
        models = [(r, gl.recommender.create(train_set,user_column='user',item_column='item',
                                                       target_column='rating',method='matrix_factorization',
                                                       n_factors=f,regularization=r,
                                                       binary_targets=True,
                                                       max_iterations=50,verbose=False), f)
                  for r in regularization_vals for f in feats]
    
        for m in models:
            print i, m[0],m[2], m[1].summary()['training_rmse'], gl.evaluation.rmse(test_set['rating'], m[1].score(test_set))
    
    return 0

def learnSentimentMatrixFactorization(trainReviews):
    logger = logging.getLogger('signature.lSMF.learnSentimentMF')
    logger.info('starting learnSentimentMatrixFactorization from %d reviews'%len(trainReviews))
    fsw = featureStructureWorker()
    modelDict = dict()
    for i, feature in enumerate(fsw.featureIdicator):
#        if feature.count('_')>1:
#            continue
        logger.debug('Start working with %s'%feature)
        if not fsw.featureIdicator[feature]:
            continue
        data = {'user':[],'item':[],'rating':[]}
        for review in trainReviews:
            reviewFeatures = fsw.getReviewFeaturesSentiment(review['features'])
            if feature not in reviewFeatures:
                continue
            
            busID = review['business_id']
            userID = review['user_id']
            rating = np.average(reviewFeatures[feature])
            
            data['user'].append(userID)
            data['item'].append(busID)
            data['rating'].append(rating)
        
        learnData = gl.SFrame(data)
        
        #CROSSS VALIDATION
        #print feature
        #getBestReg(logger, learnData)
        
        modelDict[feature] = gl.recommender.create(learnData,user_column='user',item_column='item',
                                                   target_column='rating',method='matrix_factorization',
                                                   n_factors=5,regularization=100,
                                                   #binary_targets=True,
                                                   max_iterations=50,verbose=False)
        
        logger.info('(%d) %s -> score on train: %s'%(i, feature,str(modelDict[feature].summary()['training_rmse'])))
#        if i > 5:
#            break
        #break
    return modelDict


def learnSentimentMF(path, limit = 1000000000):
    logger = logging.getLogger('signature.lSMF')
    logger.info('starting learnSentimentMF')
    #get data
    r_file = path+'/yelp_reviews_features_train.json'
    
    trainReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        trainReviews.append(json.loads(line.strip()))
    
    #run function
    modelDict = learnSentimentMatrixFactorization(trainReviews)
    
    #save model
    model_path = path+'/sentimentModels/'
    for feature in modelDict:
        modelDict[feature].save(model_path+'%s_sentiment.model'%feature)