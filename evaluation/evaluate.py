import json
import logging
import os


import numpy as np


from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

from recsys.evaluation.ranking import SpearmanRho
from recsys.evaluation.decision import PrecisionRecallF1


def preRecF1(userTestReviews,N):
    userN = {user:userTestReviews[user] for user in userTestReviews if len(userTestReviews[user])>N}
    result = dict()
    for user in userN:
        #print userN[user]
        maxGT = max([rec[0] for rec in userN[user]])
        #print maxGT
        GT_DECISION = [rec[1] for rec in userN[user] if rec[0] == maxGT]
        for predictor in range(2, len(userN[user][0])):
            listN = list()
            for rev in userN[user]:
                listN.append([rev[predictor],rev[1]])
            listN.sort(reverse = True)
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
    



def evaluateRMSE(testReviews, trainAverage, outfile):
    logger = logging.getLogger('signature.evaluate.RMSE')
    logger.info('starting evaluateRMSE from %d reviews'%len(testReviews))
    
    
    y_true = list()
    y_mf = list()
    y_signatire = [[],[],[],[]]
    y_match = list()
    
    userTestReviews = dict()
    
    for r, review in enumerate(testReviews):
        y_true.append(review['stars'])
        y_mf.append(review['MF_prediction'])
        
        userID = review['user_id']
        busID = review['business_id']
        rev = [int(review['stars']),busID, review['MF_prediction']]
        for i, x in enumerate(review['rating_prediction']):
            y_signatire[i].append(x)
            rev.append(x)
            
        rev.append(review['match'])
        y_match.append(review['match'])
        userTestReviews[userID] = userTestReviews.get(userID, [])
        userTestReviews[userID].append(rev)
    
    y_avg = [trainAverage]*len(y_true)
         
    #RMSE
    rmse_mf = np.sqrt(mean_squared_error(y_true, y_mf))
    rmse_sign = [np.sqrt(mean_squared_error(y_true, y)) for y in y_signatire]
    rmse_avg = np.sqrt(mean_squared_error(y_true, y_avg))
    outfile.write('RMSE_mf = %f\nRMSE_signature = %s\nRMSE_average = %f'%(rmse_mf, str(rmse_sign), rmse_avg))
    
    #explained_variance
    explained_variance_mf = explained_variance_score(y_true, y_mf) 
    explained_variance_sign = [explained_variance_score(y_true, y) for y in y_signatire]
    explained_variance_avg = explained_variance_score(y_true, y_avg)
    outfile.write('\n\n===============\nExplained variance regression score function\nBest possible score is 1.0, lower values are worse.')
    outfile.write('\nExplained_variance_mf = %f'%explained_variance_mf +
                  '\nExplained_variance_signature = %s'%str(explained_variance_sign) +
                  '\n''Explained_variance_average = %f'%explained_variance_avg)
    
    #correlation
    corr_mf = np.corrcoef(y_true, y_mf)[0][1]
    corr_sign = [np.corrcoef(y_true, y)[0][1] for y in y_signatire]
    corr_avg = np.corrcoef(y_true, y_avg)[0][1]
    
    y_t,y_m = [],[]
    for i,r in enumerate(y_match):
        if r>0:
            y_t.append(y_true[i])
            y_m.append(y_match[i])
    corr_match = np.corrcoef(y_true, y_match)[0][1]
    corr_m = np.corrcoef(y_t, y_m)[0][1]
    outfile.write('\n\n===============\nCorrelation_mf = %f'%corr_mf+
                  '\nCorrelation_signature = %s'%str(corr_sign)+
                  '\nCorrelation_average = %f'%corr_avg+
                  '\nCorrelation_match = %f'%corr_match+
                  '\nCorrelation_match_spec = %f on %d reviews'%(corr_m,len(y_t)))
    
    #Spearman
    mf_spearman_data = [(rat, y_mf[i]) for i,rat in enumerate(y_true)]
    mf_spearman = SpearmanRho(mf_spearman_data).compute()
    
    sign_spearman_data = [[(rat, y[i]) for i,rat in enumerate(y_true)]  for y in y_signatire]
    sign_spearman = [SpearmanRho(s).compute() for s in sign_spearman_data]
    
    avg_spearman_data = [(rat, y_avg[i]) for i,rat in enumerate(y_true)]
    avg_spearman = SpearmanRho(avg_spearman_data).compute()
    
    outfile.write('\n\n===============\nSpearman_mf = %f'%mf_spearman+
                  '\nSpearman_signature = %s'%str(sign_spearman)+
                  '\nSpearman_average = %f'%avg_spearman)
    
    
    N = 2
    l,topN = preRecF1(userTestReviews,N)
    outfile.write('\n\n===============\nTOP %d Precision,Recall,F1\nBased on %d users'%(N,l))
    for predictor in topN:
        outfile.write('\nPredictor -> %d, Decision: %s'%(predictor,str(topN[predictor])))
    
    


def evaluate(path, limit = np.Inf):
    logger = logging.getLogger('signature.evaluate')
    logger.info('starting evaluate')
    #get data
    r_file = path+'yelp_reviews_features_test_pF_sent_agg_MF_match.json'
    
    testReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        testReviews.append(json.loads(line.strip()))
    logger.info('Test Reviews loaded from %s'%r_file)
    
    trainAverage = list()
    r_file = path+'yelp_reviews_features_train.json'
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d train reviews loaded'%counter)
        trainAverage.append(json.loads(line.strip())['stars'])
    logger.info('TRAIN Reviews loaded from %s'%r_file)
    trainAverage = np.average(trainAverage)
    
    
    try:
        os.stat(path+'results/')
    except:
        os.mkdir(path+'results/')
    outfile = open(path+'/results/MF_Signature_compare.txt','w')
    #run function
    evaluateRMSE(testReviews, trainAverage, outfile)
    outfile.close()
    