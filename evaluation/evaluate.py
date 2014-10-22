import json
import logging
import os


import numpy as np


from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

from recsys.evaluation.ranking import SpearmanRho


def evaluateRMSE(testReviews):
    logger = logging.getLogger('signature.evaluate.RMSE')
    logger.info('starting evaluateRMSE from %d reviews'%len(testReviews))
    
    
    y_true = list()
    y_mf = list()
    y_signatire = list()
    
    for r, review in enumerate(testReviews):
        y_true.append(review['stars'])
        y_mf.append(review['MF_prediction'])
        y_signatire.append(review['rating_prediction'])
    
    y_avg = [np.average(y_true)]*len(y_true)
         
    #RMSE
    rmse_mf = np.sqrt(mean_squared_error(y_true, y_mf))
    rmse_sign = np.sqrt(mean_squared_error(y_true, y_signatire))
    rmse_avg = np.sqrt(mean_squared_error(y_true, y_avg))
    
    #explained_variance
    explained_variance_mf = explained_variance_score(y_true, y_mf) 
    explained_variance_sign = explained_variance_score(y_true, y_signatire)
    explained_variance_avg = explained_variance_score(y_true, y_avg)
    
    #correlation
    corr_mf = np.corrcoef(y_true, y_mf)[0][1]
    corr_sign = np.corrcoef(y_true, y_signatire)[0][1]
    corr_avg = np.corrcoef(y_true, y_avg)[0][1]
    
    #Spearman
    mf_spearman_data = [(rat, y_mf[i]) for i,rat in enumerate(y_true)]
    mf_spearman = SpearmanRho(mf_spearman_data).compute()
    
    sign_spearman_data = [(rat, y_signatire[i]) for i,rat in enumerate(y_true)]
    sign_spearman = SpearmanRho(sign_spearman_data).compute()
    
    avg_spearman_data = [(rat, y_avg[i]) for i,rat in enumerate(y_true)]
    avg_spearman = SpearmanRho(avg_spearman_data).compute()
    
    return [rmse_mf, rmse_sign, rmse_avg, explained_variance_mf, explained_variance_sign, explained_variance_avg,
            corr_mf, corr_sign, corr_avg, mf_spearman, sign_spearman, avg_spearman]


def evaluate(path, limit = np.Inf):
    logger = logging.getLogger('signature.evaluate')
    logger.info('starting evaluate')
    #get data
    r_file = path+'yelp_reviews_features_test_pF_sent_agg_MF.json'
    
    testReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        testReviews.append(json.loads(line.strip()))
    logger.info('Test Reviews loaded from %s'%r_file)
    
    
    #run function
    results = evaluateRMSE(testReviews)
    
    try:
        os.stat(path+'results/')
    except:
        os.mkdir(path+'results/')
    outfile = open(path+'/results/MF_Signature_compare.txt','w')
    outfile.write('RMSE_mf = %f\nRMSE_signature = %f\nRMSE_average = %f'%(results[0], results[1], results[2]))
    outfile.write('\n\n===============\nExplained variance regression score function\nBest possible score is 1.0, lower values are worse.')
    outfile.write('\nExplained_variance_mf = %f\nExplained_variance_signature = %f\nExplained_variance_average = %f'%(results[3], results[4],results[5]))
    outfile.write('\n\n===============\nCorrelation_mf = %f\nCorrelation_signature = %f\nCorrelation_average = %f'%(results[6], results[7],results[8]))
    outfile.write('\n\n===============\nSpearman_mf = %f\nSpearman_signature = %f\nSpearman_average = %f'%(results[9], results[10],results[11]))
    outfile.close()