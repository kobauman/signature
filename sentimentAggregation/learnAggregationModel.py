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


def encodeAspects2features(fsw, review_features):
    features_list = list()
    for aspect in fsw.featureIdicator:
        if fsw.featureIdicator[aspect] == True:
            if aspect in review_features:
                sentiment = np.average(review_features[aspect])
            else:
                sentiment = 0.0
                
            if sentiment < 0.0:
                features_list.append(-sentiment)
            else:
                features_list.append(0.0)
            
            if sentiment > 0.0:
                features_list.append(sentiment)
            else:
                features_list.append(0.0)
    return features_list


def drawPR(feature,y_true,y_pred, thres,F1, path, name = ''):
    # get (precision, recall)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
#    print precision
#    print recall
#    print thresholds
    # Create plots with pre-defined labels.
    # Alternatively, you can pass labels explicitly when calling `legend`.
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.set_title('%s, Thres = %.3f, F1 = %.3f'%(feature,thres,F1))
    ax.plot(thresholds, precision[:-1], 'k--', color = 'green', label='precision_mf')
    ax.plot(thresholds, recall[:-1], 'k:', color = 'green', label='recall_mf')
    ax.plot([thres,thres], [0, 1], "-", color = 'red')
    ax.plot([0,1], [F1, F1], "-", color = 'red')
    ax.legend(loc='lower left',shadow=True)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('value')
    
    i0 = 0
    for i,t in enumerate(thresholds):
        if t > thres:
            i0 = i
            break
    precThres,recThres = [precision[i0]],[recall[i0]]
    baselinePrecision = float(list(y_true).count(1))/len(y_true)
    #print baselinePrecision
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title('ROC curve %s, Precision = %.3f, Recall = %.3f'%(feature,precThres[0],recThres[0]))
    ax2.plot(recall, precision, 'k--', color = 'blue', label='prediction')
    ax2.plot([0, 1], [baselinePrecision,baselinePrecision],  "-", color = 'green', label='baseline')
    ax2.scatter(recThres,precThres,color='red',label='best model')
    ax2.legend(loc='upper right',shadow=True)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(path+'/modelPictures/%s_pr_%s.png'%(feature,name))
    




def learnAggregationModelsCV(trainReviews, path):
    logger = logging.getLogger('signature.lAMCV.learnAggregationModelsCV')
    logger.info('starting learnSentimentMatrixFactorization from %d reviews'%len(trainReviews))
    fsw = featureStructureWorker()
    
    learnData = list()
    learnLabels = list()
    for j, review in enumerate(trainReviews):
        reviewFeatures = fsw.getReviewFeaturesSentiment(review['features'])
        rating = review['stars']
        features = encodeAspects2features(fsw, reviewFeatures)
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
    
    #run function
    model = learnAggregationModelsCV(trainReviews, path)
    
    #save model
    pickle.dump(model,open(path+'aggregation_%d.model'%counter,'wb'))
    
    return counter
    