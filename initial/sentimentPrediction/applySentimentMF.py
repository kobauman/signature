import json
import logging
import os


import numpy as np
import graphlab

from utils.featuresStructure import featureStructureWorker



def applySentimentMF(testReviews, modelDict, featureThres, featureWeights):
    logger = logging.getLogger('signature.aSMF.applySentimentMF')
    logger.info('starting applySentimentMatrixFactorization from %d reviews'%len(testReviews))
    fsw = featureStructureWorker()
    
    
    feature_data = dict()
    reviewDict = dict()
    
    for r, review in enumerate(testReviews):
        review['predSentiments'] = dict()
        busID = review['business_id']
        userID = review['user_id']
        sentiments = fsw.getReviewFeaturesSentiment(review['features'])
        #print sentiments
        ID = busID+'###'+userID
        reviewDict[ID] = review
        for feature in review['exPredFeatures']:
            if not fsw.featureIdicator.get(feature, None):
                continue
            sentiment = np.average(sentiments.get(feature,[0.0]))
            if feature in feature_data:
                feature_data[feature]['id'].append(ID)
                feature_data[feature]['user'].append(userID)
                feature_data[feature]['item'].append(busID)
                feature_data[feature]['rating'].append(sentiment)
            else:
                feature_data[feature] = {'id':[ID],'user':[userID],'item':[busID],'rating':[sentiment]}
        if not r%1000:
            logger.debug('%d reviews processed'%r)   
    
    rmse = list()
    rmse_weighted = list()
    rmse_baseline = list()
    rmse_baseline_weighted = list()
    
    accuracy = list()
    accuracy_weighted = list()
    accuracy_baseline = list()
    accuracy_baseline_weighted = list()
    
    
    weighted_sum = list()
    for f, feature in enumerate(feature_data):
#        if f > 0:
#            break
        #print feature, feature_data[feature]
        testData = graphlab.SFrame(feature_data[feature])
        prediction = modelDict[feature].predict(testData)
        testData['prediction'] = prediction
        
        for i,ID in enumerate(testData['id']):
#            if testData['prediction'][i] == featureThres[feature]:
#                sent_pred = 0.0
#            sent_pred = 1.0 if testData['prediction'][i] > featureThres[feature] else -1.0
            sent_pred = testData['prediction'][i]# - featureThres[feature]
            reviewDict[ID]['predSentiments'][feature] = sent_pred
            
            #print reviewDict[ID]['features']
            real_sent = feature_data[feature]['rating'][i]
            #print real_sent,sent_pred, accuracy
            if real_sent*sent_pred > 0.0:
                accuracy.append(1.0)
                accuracy_weighted.append(featureWeights[feature])
            elif real_sent*sent_pred < 0.0:
                accuracy.append(0.0)
                accuracy_weighted.append(0.0)
            
            #print real_sent,sent_pred, accuracy
            
            
            if real_sent*featureThres[feature] > 0:
                accuracy_baseline.append(1.0)
                accuracy_baseline_weighted.append(featureWeights[feature])
            elif real_sent*featureThres[feature] < 0:
                accuracy_baseline.append(0.0)
                accuracy_baseline_weighted.append(0.0)
            
            
            rmse.append(pow((real_sent-sent_pred),2))
            rmse_weighted.append(pow((real_sent-sent_pred),2)*featureWeights[feature])
            rmse_baseline.append(pow((real_sent-featureThres[feature]),2))
            rmse_baseline_weighted.append(pow((real_sent-featureThres[feature]),2)*featureWeights[feature])
            weighted_sum.append(featureWeights[feature])
            
        if not f%1:
            logger.debug('%d features sentiments predicted'%f)
    
    
    #
    
    #RMSE
    rmse = np.average(rmse)
    #weighted rmse
    rmse_weighted = np.sum(rmse_weighted)/np.sum(weighted_sum)
    #rmse baseline
    rmse_baseline = np.average(rmse_baseline)
    #rmse baseline weighted
    rmse_baseline_weighted = np.sum(rmse_baseline_weighted)/np.sum(weighted_sum)
    
    
    #ACCURACY
    accuracy = np.average(accuracy)
    #weighted accuracy
    accuracy_weighted = np.sum(accuracy_weighted)/np.sum(weighted_sum)
    #accuracy baseline
    accuracy_baseline = np.average(accuracy_baseline)
    #accuracy baseline weighted
    accuracy_baseline_weighted = np.sum(accuracy_baseline_weighted)/np.sum(weighted_sum)
    
    
    #weighted accuracy
    
    return [reviewDict[i] for i in reviewDict], [rmse,rmse_weighted,rmse_baseline,rmse_baseline_weighted,
                                                 accuracy,accuracy_weighted,accuracy_baseline,accuracy_baseline_weighted]


def applySMF(path, limit = np.Inf):
    logger = logging.getLogger('signature.aSMF')
    logger.info('starting applySentimentMF')
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
    modelDict = dict()
    featureThres = dict()
    fsw = featureStructureWorker()
    for feature in fsw.featureIdicator:
        if not fsw.featureIdicator[feature]:
            continue
        try:
            modelPath = path + '/sentimentModels/%s_sentiment.model'%feature
            print modelPath
            modelDict[feature] = graphlab.load_model(modelPath)
            
            #load average
            thres_path = path+'/sentimentModels/%s_sentiment.threshold'%feature
            infile = open(thres_path,'r')
            featureThres[feature] = float(infile.readline().strip())
            infile.close()
        except:
            logger.error('There is no model for feature: %s'%feature)
            continue
        
    logger.info('Models loaded')
    
    #load featureWeights
    infile = open(path+'/featureWeights.json','r')
    featureWeights = json.loads(infile.readline().strip())
    infile.close()
    
    #run function
    reviewsPrediction, results = applySentimentMF(testReviews, modelDict, featureThres, featureWeights)
    
    #save result
    outfile = open(path+'yelp_reviews_test_predictions.json','w')
    for review in reviewsPrediction:
        outfile.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
    outfile.close()
    
    try:
        os.stat(path+'/results/')
    except:
        os.mkdir(path+'/results/')
    outfile = open(path+'/results/Sentiment_prediction.txt','w')
    outfile.write('RMSE = %f\nRMSE_weighted = %f'%(results[0], results[1]))
    outfile.write('\n\nRMSE_baseline = %f\nRMSE_baseline_weighted = %f'%(results[2], results[3]))
    outfile.write('\n===============\n\nAccuracy = %f\nAccuracy_weighted = %f'%(results[4], results[5]))
    outfile.write('\n\nAccuracy_baseline = %f\nAccuracy_baseline_weighted = %f'%(results[6], results[7]))
    outfile.close()