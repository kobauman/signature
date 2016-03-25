import json
import logging
import pickle
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from utils.featuresStructure import featureStructureWorker
from featureWorkers.getFeatures import getFeatures


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
        
    return '%.3f\t%.3f\t%.3f'%(precThres[0],recThres[0],F1)




def applyFeatureExistance(busImportantFeatures, userImportantFeatures, testReviews, modelDict, path):
    logger = logging.getLogger('signature.applyFE.aFE')
    logger.info('starting applyFeatureExistance from %d reviews'%len(testReviews))
    fsw = featureStructureWorker()
    featureWeights = dict()
    featureSWeights = dict()
    
    featureQuality = dict()
    
    for k, feature in enumerate(fsw.featureIdicator):
#        print(k,feature)
#        if k > 15:
#            break
        
        
        if not fsw.featureIdicator[feature]:
            continue
        if feature not in modelDict:
            continue
        
        logger.debug('Start working with (%d) %s'%(k,feature))
        #get data
        X1, Y1, X2, Y2, missed = getFeatures(logger, feature, testReviews, busImportantFeatures, userImportantFeatures)
        
        
        #weight = frequency
        featureWeights[feature] = float(sum(Y1))/len(Y1)
        #weight = sentiment
        featureSWeights[feature] = float(sum(Y2))/len(Y2)
        
        '''
        Existence
        '''
        #Ypred = [int(x[1] > modelDict[feature][0])  for x in modelDict[feature][1].predict_proba(np.array(X1))]
        Ypred = modelDict[feature][1].predict(np.array(X1))
        Yreal = np.array(Y1)
        
        quality = list(f1_score(Yreal, Ypred, average=None)) 
        quality += list(precision_score(Yreal, Ypred, average=None)) 
        quality += list(recall_score(Yreal, Ypred, average=None))
        
        '''
        Sentiment
        '''
        #YSpred = [int(x[1] > modelDict[feature][2]) for x in modelDict[feature][3].predict_proba(np.array(X2))]
        YSpred = modelDict[feature][3].predict(np.array(X2))
        YSreal = np.array(Y2)
        
        qualityS = list(f1_score(YSreal, YSpred, average=None)) 
        qualityS += list(precision_score(YSreal, YSpred, average=None)) 
        qualityS += list(recall_score(YSreal, YSpred, average=None))
        
        featureQuality[feature]  = [round(featureWeights[feature],2), len(Y1)]
        featureQuality[feature] += [round(x,2) for x in quality]
        featureQuality[feature] += [round(featureSWeights[feature],2), len(Y2)]
        featureQuality[feature] += [round(x,2) for x in qualityS]
        
#        print(feature,featureQuality[feature])
        
        for r, review in enumerate(testReviews):
            existence = 0
            predictedExistence = 0
            
            X1, Y1, X2, Y2, missed = getFeatures(logger, feature, [review], busImportantFeatures, userImportantFeatures)
            if len(Y1): #check if the review has enough history
                review['exPredFeatures'] = review.get('exPredFeatures', {})
            
                existence = Y1[0]
                #print Yreal[r], Ypred[r], modelDict[feature][0]
                
                
                prediction = modelDict[feature][1].predict_proba(np.array(X1))[0][1] # probability of second class!!!
                #prediction = float(modelDict[feature][1].predict(np.array(X1))[0])
                #prediction = busImportantFeatures[review['business_id']]['featureFreq'][feature]/100.0
                if prediction >= modelDict[feature][0]:
                    predictedExistence = 1
                else:
                    predictedExistence = 0
                predictedExistence = prediction
#                print(X1[0], prediction, busImportantFeatures[review['business_id']]['featureFreq'][feature]/100.0)
                randomPrediction = random.random()#int(random.random() > 0.5)
                simplePrediction = busImportantFeatures[review['business_id']]['featureFreq'][feature]/100.0#int(busImportantFeatures[review['business_id']]['featureFreq'][feature] > 40)
                basePredictionPos = 1
                basePredictionNeg = 0
                
                #print(existence, predictedExistence, randomPrediction, simplePrediction, basePredictionPos, basePredictionNeg)
                
                
                review['exPredFeatures'][feature] = [existence, predictedExistence,
                                                     randomPrediction, simplePrediction, 
                                                     basePredictionPos, basePredictionNeg]
                    
                #print(feature, review['exPredFeatures'][feature])
            
            '''
            Sentiment
            '''
            if len(Y2):
                review['sentPredFeatures'] = review.get('sentPredFeatures', {})
            
                sentiment = Y2[0]
                #print Yreal[r], Ypred[r], modelDict[feature][0]
                
                prediction = modelDict[feature][3].predict_proba(np.array(X2))[0][1]
                #prediction = float(modelDict[feature][3].predict(np.array(X2))[0])
                if prediction >= modelDict[feature][2]:
                    predictedSentiment = 1
                else:
                    predictedSentiment = 0
                predictedSentiment = prediction
                
                randomSPrediction = random.random()#int(random.random() > 0.5)
                simpleSPrediction = (busImportantFeatures[review['business_id']]['sentiment'].get(feature,[0.0,0])[0]+1)/2.0#int(busImportantFeatures[review['business_id']]['sentiment'].get(feature,[0.0,0])[0] >= -0.5)
                baseSPredictionPos = 1
                baseSPredictionNeg = 0
                
                review['sentPredFeatures'][feature] = [sentiment, predictedSentiment,
                                                       randomSPrediction, simpleSPrediction,
                                                       baseSPredictionPos, baseSPredictionNeg]
            
            if not r%5000:
                logger.debug('%d reviews processed'%r)
    
    return testReviews, featureWeights, featureQuality


def applyFE(path, modelfile, limit = np.Inf):
    logger = logging.getLogger('signature.applyFE')
    logger.info('starting applyFE')
    #get data
    b_file = path+'/businessProfile.json'
    u_file = path+'/userProfile.json'
    r_file = path+'/specific_reviews_test.json'
    
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
    modelDict = pickle.load(open(modelfile,'rb'))
    logger.info('Model loaded from %s'%modelfile)
    
    
    #run function
    reviewsPrediction, featureWeights, featureQuality = applyFeatureExistance(busImportantFeatures, userImportantFeatures,
                                                                                          testReviews, modelDict, path)
    
    #save result
    outfile = open(path+'specific_reviews_test_predictions.json','w')
    for review in reviewsPrediction:
        outfile.write(json.dumps(review)+'\n')
    outfile.close()
    
    try:
        os.stat(path+'results/')
    except:
        os.mkdir(path+'results/')
        
    #save featureWeights
    json.dump(featureWeights, open(path+'/results/featureWeights.json','w'))
    
    #save feature Quality
    outfile = open(path+'/results/featureQuality.txt','w')
    feats = list(featureQuality.keys())
    feats.sort()
    outfile.write('Feature_name\texistAcc\tlen(Y1)\tsentAcc\tlen(Y2)\tmissed\n')
    for feat in feats:
        outfile.write('%20s\t%s\n'%(feat,'\t'.join([str(x) for x in featureQuality[feat]])))
    outfile.close()
