import json
import logging
import os


import numpy as np


from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

from recsys.evaluation.ranking import SpearmanRho
from recsys.evaluation.decision import PrecisionRecallF1


def preRecF1(userTestReviews,N):
    #get users with long history
    userN = {user:userTestReviews[user] for user in userTestReviews if len(userTestReviews[user])>N}
    #predictor: result
    result = dict()
    #iterate by users
    for user in userN:
        #print userN[user]
        maxGT = max([rec[0] for rec in userN[user]])
        #print maxGT
        GT_DECISION = [rec[1] for rec in userN[user] if rec[0] == maxGT]
        for predictor in userN[user][0][2]:
            listN = list()
            for rev in userN[user]:
                listN.append([rev[2][predictor],rev[1]])
            listN.sort(reverse = True)
            #get first N recommendations
            TEST_DECISION = [rec[1] for rec in listN[:N]]
            #print 'TEST_DECISION',TEST_DECISION
            decision = PrecisionRecallF1()
            decision.load(GT_DECISION, TEST_DECISION)
            destuple = decision.compute()
            
            result[predictor] = result.get(predictor, [[],[],[]])
            for i in range(len(destuple)):
                result[predictor][i].append(destuple[i])
    for predictor in result:
        for i in range(len(result[predictor])):
            result[predictor][i] = round(np.average(result[predictor][i]),3)
    return len(userN),result
    



def evaluateRMSE(testReviews, trainAverage, userImportantFeatures,
                 outfile, userProfileThres = 0):
    
    logger = logging.getLogger('signature.evaluate.RMSE')
    logger.info('starting evaluateRMSE from %d reviews'%len(testReviews))
    
    
    
    userTestReviews = dict()
    
    for review in testReviews:    
        userID = review['user_id']
        userTestReviews[userID] = userTestReviews.get(userID, [])
        userTestReviews[userID].append(1)
        
    userGoodSet = set([user for user in userTestReviews if len(userTestReviews[user])>userProfileThres and user in userImportantFeatures])
    y_true = list()
    y_prediction = dict()
    y_prediction['baseline_avg'] = [[],[]]
    
    userTestReviews = dict()
    
    for r, review in enumerate(testReviews):
        if review['user_id'] not in userGoodSet:
            continue
        y_true.append(review['stars'])
        for predictor in review['rating_prediction']:
            y_prediction[predictor] = y_prediction.get(predictor,[[],[]])
            y_prediction[predictor][0].append(review['stars'])
            prediction = review['rating_prediction'][predictor]
            if not prediction:
                prediction = trainAverage
            y_prediction[predictor][1].append(prediction)
        
        y_prediction['baseline_avg'][0].append(review['stars'])
        y_prediction['baseline_avg'][1].append(trainAverage)
        
        userID = review['user_id']
        busID = review['business_id']
        userTestReviews[userID] = userTestReviews.get(userID, [])
        userTestReviews[userID].append([review['stars'],busID,review['rating_prediction']])
    
    
    
    #RMSE
    outfile.write('===============\nRMSE\n')
    for predictor in y_prediction:
        y_true = y_prediction[predictor][0]
        y_pred = y_prediction[predictor][1]
        #print len(y_true), len(y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        outfile.write('%s : \t %f\n'%(predictor,rmse))
    
    #explained_variance
    outfile.write('\n\n===============\nExplained variance regression score function\nBest possible score is 1.0, lower values are worse.\n')
    for predictor in y_prediction:
        y_true = y_prediction[predictor][0]
        y_pred = y_prediction[predictor][1]
        ev = explained_variance_score(y_true, y_pred)
        outfile.write('%s : \t %f\n'%(predictor,ev))
        

    #correlation
    outfile.write('\n\n===============\nCorrelation\n')
    for predictor in y_prediction:
        y_true = y_prediction[predictor][0]
        y_pred = y_prediction[predictor][1]
        corr = np.corrcoef(y_true, y_pred)[0][1]
        outfile.write('%s : \t %f\n'%(predictor,corr))
    
    #Spearman
    outfile.write('\n\n===============\nSpearman\n')
    for predictor in y_prediction:
        y_true = y_prediction[predictor][0]
        y_pred = y_prediction[predictor][1]
        spearman_data = [(rat, y_pred[i]) for i,rat in enumerate(y_true)]
        spearman = SpearmanRho(spearman_data).compute()
        outfile.write('%s : \t %f\n'%(predictor,spearman))
    
    
    
    Ns = [2,5,10]
    for N in Ns:
        l,topN = preRecF1(userTestReviews,N)
        outfile.write('\n\n===============\nTOP %d \t[Precision,Recall,F1]\nBased on %d users\n'%(N,l))
        for predictor in topN:
            outfile.write('%s: \t %s\n'%(predictor,str(topN[predictor])))
            
   
    
#    
#    N = 5
#    l,topN = preRecF1(userTestReviews,N)
#    outfile.write('\n\n===============\nTOP %d Precision,Recall,F1\nBased on %d users'%(N,l))
#    for predictor in topN:
#        outfile.write('\nPredictor -> %d, Decision: %s'%(predictor,str(topN[predictor])))
#    
#    
#    N = 10
#    l,topN = preRecF1(userTestReviews,N)
#    outfile.write('\n\n===============\nTOP %d Precision,Recall,F1\nBased on %d users'%(N,l))
#    for predictor in topN:
#        outfile.write('\nPredictor -> %d, Decision: %s'%(predictor,str(topN[predictor])))
    

def evaluate(path, userProfileThres = 0, limit = np.Inf):
    logger = logging.getLogger('signature.evaluate')
    logger.info('starting evaluate')
    #get data
    test_file = path+'yelp_reviews_test_predictions.json'
    
    #load user profiles
    u_file = path+'userFeaturesAggregation_stat.json'
    userImportantFeatures = json.loads(open(u_file,'r').readline())
    logger.info('Important USER Features loaded')
    
    #get train Average
    trainAverage = list()
    r_file = path+'yelp_reviews_features_train.json'
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d train reviews loaded'%counter)
        trainAverage.append(json.loads(line.strip())['stars'])
    logger.info('TRAIN Reviews loaded from %s'%r_file)
    trainAverage = np.average(trainAverage)
    
    #load test data
    testReviews = list()
    for counter, line in enumerate(open(test_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        testReviews.append(json.loads(line.strip()))
    logger.info('Test Reviews loaded from %s'%test_file)
    
    
    try:
        os.stat(path+'results/')
    except:
        os.mkdir(path+'results/')
    outfile = open(path+'/results/MF_Signature_compare.txt','w')
    #run function
    evaluateRMSE(testReviews, trainAverage, userImportantFeatures, outfile, userProfileThres)
    outfile.close()
    