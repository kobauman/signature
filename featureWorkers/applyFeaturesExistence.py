import json
import logging
import pickle
import random
import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

from utils.featuresStructure import featureStructureWorker
from getFeatures import getFeatures


def drawPR(feature,y_true,y_pred,y_bus,thres, path):
    # get (precision, recall)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    #random prediction
    y_rand = [random.random() for i in range(len(y_pred))]
    precision_rand, recall_rand, thresholds_rand = precision_recall_curve(y_true, y_rand)
    #bus prediction
    precision_bus, recall_bus, thresholds_bus = precision_recall_curve(y_true, y_bus)
    # Create plots with pre-defined labels.
    # Alternatively, you can pass labels explicitly when calling `legend`.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    i0 = 0
    for i,t in enumerate(thresholds):
        if t > thres:
            i0 = i
            break
    precThres,recThres = [precision[i0]],[recall[i0]]
    F1 = 2*recThres[0]*precThres[0]/(recThres[0]+precThres[0])
    baselinePrecision = float(list(y_true).count(1))/len(y_true)
    #print baselinePrecision
    ax.set_title('ROC curve %s, \nF1 = %.3f, Recall = %.3f, Precision = %.3f, Percent = %.3f'%(feature,
                                                                                               F1,recThres[0],
                                                                                               precThres[0],
                                                                                               baselinePrecision))
    ax.plot(recall[:-1], precision[:-1], 'k--', color = 'blue', label='prediction')
    ax.plot(recall_rand[:-1], precision_rand[:-1], 'k--', color = 'purple', label='random')
    ax.plot(recall_bus[:-1], precision_bus[:-1], 'k--', color = 'grey', label='business')
    ax.plot([0, 1], [baselinePrecision,baselinePrecision],  "-", color = 'green', label='baseline')
    ax.scatter(recThres,precThres,color='red',label='applied model')
    ax.legend(loc='upper right',shadow=True)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    
    #plt.subplots_adjust(hspace=0.5)
    try:
        os.stat(path+'/testPictures/')
    except:
        os.mkdir(path+'/testPictures/')
    plt.savefig(path+'/testPictures/%s.png'%feature)
        
    return F1




def applyFeatureExistance(busImportantFeatures, userImportantFeatures, testReviews, modelDict, trainAveragesDict, path):
    logger = logging.getLogger('signature.aFE.applyFE')
    logger.info('starting applyFeatureExistance from %d reviews'%len(testReviews))
    fsw = featureStructureWorker()
    featureWeights = dict()
    featureF1 = dict()
    
    for i, feature in enumerate(fsw.featureIdicator):
        if not fsw.featureIdicator[feature]:
            continue
        logger.debug('Start working with %s'%feature)
        #get data
        X, Y = getFeatures(logger, feature, testReviews, busImportantFeatures, userImportantFeatures,
                                          trainAverages = trainAveragesDict[feature], is_train = False)
        
        #weight = frequency
        featureWeights[feature] = float(list(Y).count(1))/len(Y)
        
        Ypred = [x[1] for x in modelDict[feature][2].predict_proba(np.array(X))]
        Yreal = np.array(Y)
        
        Ybus = []
        for review in testReviews:
            busID = review['business_id']
            if busID in busImportantFeatures:
                pfreq = busImportantFeatures[busID]['featureFreq'].get(feature,0.0)
            else:
                pfreq = featureWeights[feature]
            Ybus.append(pfreq)
        
        featureF1[feature] = drawPR(feature,Yreal,Ypred,Ybus, modelDict[feature][0], path)
        
        for r, review in enumerate(testReviews):
            #reviewFeatures = fsw.getReviewFeaturesExistence(review['features'])
            review['exPredFeatures'] = review.get('exPredFeatures', {})
        
            existence = Yreal[r]
            #print Yreal[r], Ypred[r], modelDict[feature][0]
            if Ypred[r] >= modelDict[feature][0]:
                predictedExistence = 1
            else:
                predictedExistence = 0
                
            #check if feature important
            if existence + predictedExistence > 0.5:
                review['exPredFeatures'][feature] = [existence, predictedExistence]
                
            #print review['exPredFeatures']
            if not r%10000:
                logger.debug('%d reviews processed'%r)
        
    Jaccard = list()
    Jaccard_weighted = list()
    Jaccard_baseline = list()
    Jaccard_baseline_weighted = list()
    for r, review in enumerate(testReviews):
        Jaccard_intersection = 0.0
        Jaccard_union = 0.0
        
        Jaccard_intersection_weighted = 0.0
        Jaccard_union_weighted = 0.0
        
        Jaccard_intersection_baseline = 0.0
        Jaccard_union_baseline = 0.0
        
        Jaccard_intersection_baseline_weighted = 0.0
        Jaccard_union_baseline_weighted = 0.0
        
        for feature in review['exPredFeatures']:
            if review['exPredFeatures'][feature] == [1,1]:
                Jaccard_intersection += 1.0
                Jaccard_intersection_weighted += featureWeights[feature]
            Jaccard_union += 1.0
            Jaccard_union_weighted += featureWeights[feature]
            
            if review['exPredFeatures'][feature][0] == 1:
                Jaccard_intersection_baseline  += 1.0
                Jaccard_intersection_baseline_weighted += featureWeights[feature]
        
        for feature in fsw.featureIdicator:
            if fsw.featureIdicator[feature]:
                Jaccard_union_baseline += 1
                Jaccard_union_baseline_weighted += featureWeights[feature]
        
        if Jaccard_union:
            Jaccard.append(Jaccard_intersection/Jaccard_union)       
        if Jaccard_union_weighted:
            Jaccard_weighted.append(Jaccard_intersection_weighted/Jaccard_union_weighted)
        if Jaccard_union_baseline:
            Jaccard_baseline.append(Jaccard_intersection_baseline/Jaccard_union_baseline)
        if Jaccard_union_baseline_weighted:
            Jaccard_baseline_weighted.append(Jaccard_intersection_baseline_weighted/Jaccard_union_baseline_weighted)
        
    
    
    return testReviews, featureWeights, [np.average(Jaccard), np.average(Jaccard_weighted),
                         np.average(Jaccard_baseline), np.average(Jaccard_baseline_weighted)], featureF1


def applyFE(path, modelfile, trainAveragesFile, limit = np.Inf):
    logger = logging.getLogger('signature.aFE')
    logger.info('starting applyFE')
    #get data
    b_file = path+'/businessFeaturesAggregation_train.json'
    u_file = path+'/userFeaturesAggregation_train.json'
    r_file = path+'/yelp_reviews_features_test.json'
    
    busImportantFeatures = json.loads(open(b_file,'r').readline())
    logger.info('Important BUSINESS Features loaded')
    userImportantFeatures = json.loads(open(u_file,'r').readline())
    logger.info('Important USER Features loaded')
    testReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        testReviews.append(json.loads(line.strip()))
    logger.info('Reviews loaded')
    
    #load model
    modelDict = pickle.load(open(modelfile,'r'))
    logger.info('Model loaded from %s'%modelfile)
    
    #load averages
    infile = open(trainAveragesFile,'rb')
    trainAveragesDict = json.loads(infile.read())
    infile.close()
    
    #run function
    reviewsPrediction, featureWeights, Jaccard, featureF1 = applyFeatureExistance(busImportantFeatures, userImportantFeatures,
                                                                       testReviews, modelDict, trainAveragesDict,
                                                                       path)
    
    #save result
    outfile = open(path+'/yelp_reviews_features_test_pF.json','w')
    for review in reviewsPrediction:
        outfile.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
    outfile.close()
    
    
    #save featureWeights
    outfile = open(path+'/featureWeights.json','w')
    outfile.write(json.dumps(featureWeights).encode('utf8', 'ignore'))
    outfile.close()
    
    try:
        os.stat(path+'/results/')
    except:
        os.mkdir(path+'/results/')
    outfile = open(path+'/results/Jaccard_feature_existence.txt','w')
    outfile.write('Jaccard = %f\nJaccard_weighted = %f'%(Jaccard[0], Jaccard[1]))
    outfile.write('\n\nJaccard_baseline = %f\nJaccard_baseline_weighted = %f'%(Jaccard[2], Jaccard[3]))
    outfile.close()
    
    outfile = open(path+'/results/FeatureExistenceF1.txt','w')
    outfile.write('\n'.join(['%s: \t%.3f'%(feature,featureF1[feature]) for feature in featureF1]))
    outfile.close()