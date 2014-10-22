import json
import logging
import random

import numpy as np
import graphlab
import matplotlib.pyplot as plt

from utils.featuresStructure import featureStructureWorker
from utils.precision import precision_recall_curve



    



def getBestMFThres(logger, feature, learnData, path):
#    threses = list()
#    qualities = list()
#    for counter in range(3):
#        train, test = learnData.random_split(0.8)
#        
#        MFmodel = graphlab.recommender.factorization_recommender.create(train,user_id='user',item_id='item',
#                                                                  target='rating',num_factors=10,
#                                                                  #regularization=100,#binary_target=True,
#                                                                  max_iterations=50,verbose=False)
#        
##        MFmodel = graphlab.recommender.item_similarity_recommender.create(train,user_id='user',item_id='item',
##                                                                          target='rating', similarity_type='jaccard',
##                                                                          training_method='auto', threshold=0.001, 
##                                                                          only_top_k=100, random_seed=0, verbose=True)
#        test_predictions = MFmodel.predict(test)
#        print 'train average', np.average(train['rating'])
#        print 'test average', np.average(test['rating'])
#        print test_predictions#[:5]
#        print test['rating']#[:5]
##        print graphlab.evaluation.accuracy(test['rating'], test_predictions)
#        
#        avg = 0
#        predictions = list()
#        for i in range(len(test)):
#            pred = -1 if test_predictions[i] < avg else 1
#            predictions.append(pred)
#        print graphlab.evaluation.accuracy(graphlab.SArray(test['rating']), graphlab.SArray(predictions))
#        print graphlab.evaluation.confusion_matrix(graphlab.SArray(test['rating']), graphlab.SArray(predictions))
#        
#        
#        
#        avg = np.average(train['rating'])
#        #avg = 0
#        predictions = list()
#        for i in range(len(test)):
#            pred = -1 if test_predictions[i] < avg else 1
#            predictions.append(pred)
#        print graphlab.evaluation.accuracy(graphlab.SArray(test['rating']), graphlab.SArray(predictions))
#        print graphlab.evaluation.confusion_matrix(graphlab.SArray(test['rating']), graphlab.SArray(predictions))
##        exit()
##        print test['rating'], test_predictions
#        prec, rec, thresholds = precision_recall_curve(-np.array(test['rating']), -np.array(test_predictions))
##        prec_minus, rec_minus, thresholds_minus = precision_recall_curve(-np.array(test['rating']), -np.array(test_predictions))
##        prec_minus = prec_minus[::-1]
##        rec_minus = rec_minus[::-1]
#        #print prec, rec, thresholds
##        print 'thresholds',thresholds,-thresholds_minus[::-1]
#        bestF1 = 0.0
#        bestThres = 0.0
#        
#        for i, thres in enumerate(thresholds[:-1]):
#            #print thres, prec[i],rec[i]
#            try:
#                F1 = 2*prec[i]*rec[i]/(prec[i]+rec[i])
##                F1_minus = 2*prec_minus[i]*rec_minus[i]/(prec_minus[i]+rec_minus[i])
##                F1 = 2*F1_plus*F1_minus/(F1_plus+F1_minus)
#            except:
#                continue
#            #print F1, prec[i], rec[i]
#            if F1 > bestF1:
#                bestF1 = F1
#                bestThres = -thres
#        print '=============='
#        print counter, bestF1, bestThres
##        logger.info('Best score on LR with thres = %.3f is equal: %.3f'%(bestThres,bestF1))
##        #drawPR(feature,testY,Ypred,bestThres,bestF1, path, 'LR_%d'%counter)
#        threses.append(bestThres)
#        qualities.append(bestF1)
#        
#        
#        
#        
#        predictions = list()
#        for i in range(len(test)):
#            pred = -1 if test_predictions[i] < bestThres else 1
#            predictions.append(pred)
#        print graphlab.evaluation.accuracy(graphlab.SArray(test['rating']), graphlab.SArray(predictions))
#        print graphlab.evaluation.confusion_matrix(graphlab.SArray(test['rating']), graphlab.SArray(predictions))
    
    
    MFmodel = graphlab.recommender.factorization_recommender.create(learnData,user_id='user',item_id='item',
                                                                             target='rating',num_factors=10,
                                                                             #regularization=100,#binary_targets=True,
                                                                             max_iterations=50,verbose=False)
    
    
    
    bestThres = np.average(learnData['rating'])#threses)
    #bestF1 = np.average(qualities)
    #get quality on test!!!!!
    return bestThres,MFmodel



def learnSentimentMatrixFactorization(trainReviews, path):
    logger = logging.getLogger('signature.lSMF.learnSentimentMF')
    logger.info('starting learnSentimentMatrixFactorization from %d reviews'%len(trainReviews))
    fsw = featureStructureWorker()
    modelDict = dict()
    featureThres = dict()
    for i, feature in enumerate(fsw.featureIdicator):
#        if feature.count('_')>1:
#            continue
        logger.debug('Start working with %s'%feature)
        
        if not fsw.featureIdicator[feature]:
            continue
        
        learnData = {'user':[],'item':[],'rating':[]}
        
        for j, review in enumerate(trainReviews):
            reviewFeatures = fsw.getReviewFeaturesSentiment(review['features'])
            if feature not in reviewFeatures:
                continue
            
            busID = review['business_id']
            userID = review['user_id']
            rating = np.average(reviewFeatures[feature])
#            if rating == 0.0:
#                continue
#            rating = 1.0 if rating > 0 else -1.0
            learnData['user'].append(userID)
            learnData['item'].append(busID)
            learnData['rating'].append(rating)
            
            
        #CROSSS VALIDATION
        data = graphlab.SFrame(learnData)
        featureThres[feature], modelDict[feature] = getBestMFThres(logger, feature, data, path)

    
    return modelDict, featureThres


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
    modelDict, featureThres = learnSentimentMatrixFactorization(trainReviews, path)
    
    #save model
    model_path = path+'/sentimentModels/'
    for feature in modelDict:
        modelDict[feature].save(model_path+'%s_sentiment.model'%feature)
    
    #save average
    model_path = path+'/sentimentModels/'
    for feature in featureThres:
        output = open(model_path+'%s_sentiment.threshold'%feature,'w')
        output.write(str(featureThres[feature]))
        output.close()