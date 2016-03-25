import json
import logging
import random
import os

import numpy as np
import graphlab
import matplotlib.pyplot as plt

from utils.featuresStructure import featureStructureWorker
from utils.precision import precision_recall_curve



def getBestMFThres(logger, feature, learnData, path):
    bestWin = 0.0
    bestReg = 0.01
    
#    for reg in [0.01,0.1]:
#        train, test = learnData.random_split(0.8)
#        print(len(train),len(test))
#        MFmodel = graphlab.recommender.factorization_recommender.create(train,user_id='user',item_id='item',
#                                                                  target='rating',num_factors=50,
#                                                                  regularization=reg,binary_target=True,
#                                                                  max_iterations=50,verbose=False)
#        
#
#        print('Regularization = %f'%reg)
#        test_predictions = MFmodel.predict(test)
##        print 'train average', np.average(train['rating'])
##        print 'test average', np.average(test['rating'])
##        print 'RMSE average(TEST) = ', graphlab.evaluation.rmse(graphlab.SArray(test['rating']), graphlab.SArray([np.average(test['rating'])]*len(test['rating'])))
#        rmse_avg = graphlab.evaluation.rmse(graphlab.SArray(test['rating']), graphlab.SArray([np.average(train['rating'])]*len(test['rating'])))
#        print('RMSE average = ', rmse_avg)
#        rmse_predict = graphlab.evaluation.rmse(graphlab.SArray(test['rating']), test_predictions)
#        print('RMSE predict = ', rmse_predict)
#        win = (rmse_avg - rmse_predict)/rmse_avg
#        print('RMSE average - predict = ',win)
#        if win > bestWin:
#            bestWin = win
#            bestReg = reg
#
#    
#    print('Best WIN = ',bestWin,'With reg = ',bestReg)
    
    
    MFmodel = graphlab.recommender.factorization_recommender.create(learnData,user_id='user',item_id='item',
                                                                    target='rating',num_factors=50,
                                                                    regularization=bestReg,binary_target=True,
                                                                    max_iterations=100,verbose=False)
    
    
    
    bestThres = np.average(learnData['rating'])#threses)
    #bestF1 = np.average(qualities)
    #get quality on test!!!!!
    return bestThres,MFmodel



def learnSentimentMatrixFactorization(trainReviews, path):
    logger = logging.getLogger('signature.learnSentimentMF.Worker')
    logger.info('starting learnSentimentMatrixFactorization from %d reviews'%len(trainReviews))
    fsw = featureStructureWorker()
    modelDict_ex = dict()
    featureThres_ex = dict()
    
    modelDict = dict()
    featureThres = dict()
    for i, feature in enumerate(fsw.featureIdicator):
#        if feature != 'SERVICE':
#            continue
        if not fsw.featureIdicator[feature]:
            continue
        
        logger.debug('Start working with (%d) %s'%(i,feature))
        
        learnData_ex = {'user':[],'item':[],'rating':[]}
        learnData = {'user':[],'item':[],'rating':[]}
        
        
        
        for j, review in enumerate(trainReviews):
            reviewFeatures = fsw.getReviewFeaturesSentiment(review['features'])
            
            
            busID = review['business_id']
            userID = review['user_id']
            
            learnData_ex['user'].append(userID)
            learnData_ex['item'].append(busID)
            if feature in reviewFeatures:
                learnData_ex['rating'].append(1)
            else:
                learnData_ex['rating'].append(0)
                
                
            if feature not in reviewFeatures:
                continue   
            
            sent = np.average(reviewFeatures[feature])
            if sent:
                learnData['user'].append(userID)
                learnData['item'].append(busID)
                
                if sent > 0:
                    learnData['rating'].append(1)
                elif sent < 0:
                    learnData['rating'].append(0)
            
            
        if len(learnData_ex['rating']):
            data_ex = graphlab.SFrame(learnData_ex)
            featureThres_ex[feature], modelDict_ex[feature] = getBestMFThres(logger, feature, data_ex, path)

           
        #CROSSS VALIDATION
        if len(learnData['rating']):
            data = graphlab.SFrame(learnData)
            featureThres[feature], modelDict[feature] = getBestMFThres(logger, feature, data, path)

    return modelDict_ex, featureThres_ex, modelDict, featureThres


def learnSentimentMF(path, limit = 1000000000):
    logger = logging.getLogger('signature.learnSentimentMF')
    logger.info('starting learnSentimentMF')
    #get data
    r_file = path+'/specific_reviews_train.json'
    
    trainReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%5000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        trainReviews.append(json.loads(line.strip()))
    
    #run function
    modelDict_ex, featureThres_ex, modelDict, featureThres = learnSentimentMatrixFactorization(trainReviews, path)
    
    try:
        os.stat(path+'sentimentModels/')
    except:
        os.mkdir(path+'sentimentModels/')
    
    #save model
    model_path = path+'sentimentModels/'
    for feature in modelDict_ex:
        modelDict_ex[feature].save(model_path+'%s_sentiment_ex.model'%feature)
    
    #save average
    model_path = path+'sentimentModels/'
    for feature in featureThres_ex:
        output = open(model_path+'%s_sentiment_ex.threshold'%feature,'w')
        output.write(str(featureThres_ex[feature]))
        output.close()
    
        
    #save model
    model_path = path+'sentimentModels/'
    for feature in modelDict:
        modelDict[feature].save(model_path+'%s_sentiment.model'%feature)
    
    #save average
    model_path = path+'sentimentModels/'
    for feature in featureThres:
        output = open(model_path+'%s_sentiment.threshold'%feature,'w')
        output.write(str(featureThres[feature]))
        output.close()
    
#    output = open(model_path+'trainSentimentAverages.json','wb')
#    output.write(json.dumps(featureThres).encode('utf8', 'ignore'))
#    output.close()