import json
import logging
import pickle
import random
import os
#import graphlab as gl
import matplotlib.pyplot as plt

import numpy as np
#from sklearn.metrics import precision_recall_curve
#from sklearn.externals.six import StringIO
#from sklearn.cross_validation import KFold
#from sklearn import cross_validation
#from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


import sys
sys.path.append('../')
from utils.featuresStructure import featureStructureWorker
from featureWorkers.getFeatures import getFeatures


def getLogModel(logger, feature, X, Y, path):
#    threses = list()
#    qualities = list()
    
    CF1 = 0.0
    bestDepth = 0.5
    bestQ = [0.0,0.0]
    #for C in [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]:
    for depth in [3,5,7,10,20,30]:
    
        indicator = [i for i in range(len(X))]
        random.shuffle(indicator)
        thres = int(len(indicator)*0.8)
        trainX = np.array([X[i] for i in indicator[:thres]])
        trainY = np.array([Y[i] for i in indicator[:thres]])
        testX = np.array([X[i] for i in indicator[thres:]])
        testY = np.array([Y[i] for i in indicator[thres:]])
        #logmodel = linear_model.LogisticRegression(penalty='l1',C=C, class_weight='auto').fit(trainX, trainY)
        #logmodel = LinearSVC(C=C, class_weight='auto').fit(trainX, trainY)
        #logmodel = Ridge(alpha=C,fit_intercept=True).fit(trainX, trainY)
        logmodel = ExtraTreesClassifier(n_estimators=250, max_depth = depth, class_weight='auto').fit(trainX, trainY)
#        print(depth, logmodel.feature_importances_) 
        Ypred = [int(x[1] > 0.5) for x in logmodel.predict_proba(testX)]
        #Ypred = [int(x > 0.5) for x in logmodel.predict(testX)]
        #Ypred = logmodel.predict(testX)
        
        #logger.debug(str(C)+'\t'+str(f1_score(testY, Ypred, average=None)))
        f1 = f1_score(testY, Ypred, average=None)
        print(f1)
        if len(f1) == 1:
            avgf1 = 0.0
        else:
            avgf1 = 2*f1[0]*f1[1]/(f1[0]+f1[1])
        
#        th = np.percentile(logmodel.feature_importances_,20)
#        Xnew = logmodel.transform(trainX,th)
#        logmodel_new = ExtraTreesClassifier(n_estimators=250, max_depth = depth, class_weight='auto').fit(Xnew, trainY)
#        testXnew = logmodel.transform(testX,th)
#        Ypred_new = [int(x[1] > 0.5) for x in logmodel_new.predict_proba(testXnew)]
#        f1_new = f1_score(testY, Ypred_new, average=None)
#        
#        print(f1, f1_new)
        
        
        
        
        
        quality = list(f1_score(testY, Ypred, average=None)) 
        quality += list(precision_score(testY, Ypred, average=None)) 
        quality += list(recall_score(testY, Ypred, average=None))
        print(avgf1, f1)
        if avgf1 > CF1:
            bestDepth = depth
            CF1 = avgf1
            bestQ = [round(x,2) for x in quality]
    
    
    logger.info('%s: best Depth = %.5f with f1 = %.2f,%.2f'%(feature,bestDepth, bestQ[0],bestQ[1]))
        
    X = np.array(X)
    Y = np.array(Y)
    #logmodel = linear_model.LogisticRegression(penalty='l1',C=bestC, class_weight='auto').fit(X, Y)
    #logmodel = LinearSVC(C=bestC, class_weight='auto').fit(X, Y)
    #logmodel = Ridge(alpha=bestC,fit_intercept=True).fit(X, Y)
    
    logmodel = ExtraTreesClassifier(n_estimators=250, max_depth = bestDepth, class_weight='auto').fit(X, Y)
#    print(logmodel.coef_)
    #bestThres = np.average(threses)
    #bestF1 = np.average(qualities)
    #get quality on test!!!!!
    
    print(logmodel.feature_importances_) 
        
    bestThres = 0.5
    return bestThres,bestQ,logmodel


        

#def drawPR(feature,y_true,y_pred, thres,F1, path, name = ''):
#    # get (precision, recall)
#    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
##    print precision
##    print recall
##    print thresholds
#    # Create plots with pre-defined labels.
#    # Alternatively, you can pass labels explicitly when calling `legend`.
#    fig = plt.figure()
#    ax = fig.add_subplot(2, 1, 1)
#    ax.set_title('%s, Thres = %.3f, F1 = %.3f'%(feature,thres,F1))
#    ax.plot(thresholds, precision[:-1], 'k--', color = 'green', label='precision_mf')
#    ax.plot(thresholds, recall[:-1], 'k:', color = 'green', label='recall_mf')
#    ax.plot([thres,thres], [0, 1], "-", color = 'red')
#    ax.plot([0,1], [F1, F1], "-", color = 'red')
#    ax.legend(loc='lower left',shadow=True)
#    ax.set_xlabel('Threshold')
#    ax.set_ylabel('value')
#    
#    i0 = 0
#    for i,t in enumerate(thresholds):
#        if t > thres:
#            i0 = i
#            break
#    precThres,recThres = [precision[i0]],[recall[i0]]
#    baselinePrecision = float(list(y_true).count(1))/len(y_true)
#    #print baselinePrecision
#    ax2 = fig.add_subplot(2, 1, 2)
#    ax2.set_title('ROC curve %s, Precision = %.3f, Recall = %.3f'%(feature,precThres[0],recThres[0]))
#    ax2.plot(recall, precision, 'k--', color = 'blue', label='prediction')
#    ax2.plot([0, 1], [baselinePrecision,baselinePrecision],  "-", color = 'green', label='baseline')
#    ax2.scatter(recThres,precThres,color='red',label='best model')
#    ax2.legend(loc='upper right',shadow=True)
#    ax2.set_xlabel('Recall')
#    ax2.set_ylabel('Precision')
#    
#    plt.subplots_adjust(hspace=0.5)
#    try:
#        os.stat(path+'modelPictures/')
#    except:
#        os.mkdir(path+'modelPictures/')
#    plt.savefig(path+'modelPictures/%s_pr_%s.png'%(feature,name))
    
    

def learnFeatureExistance(busImportantFeatures, userImportantFeatures, trainReviews, path):
    logger = logging.getLogger('signature.learnFE')
    logger.info('starting learnFeatureExistance from %d reviews'%len(trainReviews))
    fsw = featureStructureWorker()
    modelDict = dict()
    
    missed_prediction = dict()
    for f, feature in enumerate(fsw.featureIdicator):
        if not fsw.featureIdicator[feature]:
            continue
        logger.info('Start working with (%d) %s'%(f,feature))
        #get data
        X1, Y1, X2, Y2, missed = getFeatures(logger, feature, trainReviews, busImportantFeatures, userImportantFeatures)
        missed_prediction[feature] = [missed, len(Y1)]
        
        
#        stat_line = '%d (%d/%d)reviews (%d of them pos(%d)/neg(%d))'%(len(Y1),sum(Y1),len(Y1) - sum(Y1),
#                                                                      len(Y2),sum(Y2),len(Y2) - sum(Y2))

        logger.debug('Got features for %d (%d/%d)reviews (%d of them pos(%d)/neg(%d))'%(len(Y1),sum(Y1),len(Y1) - sum(Y1),
                                                                                     len(Y2),sum(Y2),len(Y2) - sum(Y2)))

        print(len(Y1),len(Y2))
        if len(Y1) < 100 or sum(Y1) < 50 or len(Y1) - sum(Y1) < 50:
            continue
        if len(Y2) < 100 or sum(Y2) < 50 or len(Y2) - sum(Y2) < 50:
            continue

#        if len(Y1) < 10 or sum(Y1) < 10 or len(Y1) - sum(Y1) < 10:
#            continue
#        if len(Y2) < 10 or sum(Y2) < 10 or len(Y2) - sum(Y2) < 10:
#            continue


#        #cross validation
#        indicator = range(len(X))
#        random.shuffle(indicator)
#        thres = int(len(indicator)*0.8)
#        trainX = np.array([X[i] for i in indicator[:thres]])
#        trainY = np.array([Y[i] for i in indicator[:thres]])
#        testX = np.array([X[i] for i in indicator[thres:]])
#        testY = np.array([Y[i] for i in indicator[thres:]])
        
        #Logistic Regression
        bestThres, bestQ,logmodel = getLogModel(logger, feature, X1, Y1, path)
        
        logger.info('Sentiment prediction for (%d) %s'%(f,feature))
        #Logistic Regression
        bestThres_2, bestQ_2, logmodel_2 = getLogModel(logger, feature, X2, Y2, path)
        
        
        feat_info = [len(Y1), sum(Y1), len(Y1) - sum(Y1)] + bestQ + [len(Y2), sum(Y2),len(Y2) - sum(Y2)] + bestQ_2
        
        #bestThresSVM,bestF1SVM,svmmodel = getBestSVMModel(logger, feature, X, Y, path)
        
#       crossValidation(logger, np.array(X), np.array(Y))
        
        
        modelDict[feature] = [bestThres, logmodel, bestThres_2, logmodel_2, feat_info]
        
#        print(f)
#        if f > 6:
#            break
        
    return modelDict


def learnFE(path, limit = np.Inf):
    logger = logging.getLogger('signature.learnFE')
    logger.info('starting learnFE')
    #get data
    b_file = path+'/businessProfile.json'
    u_file = path+'/userProfile.json'
    r_file = path+'/specific_reviews_extrain.json'
    
    busImportantFeatures = json.loads(open(b_file,'r').readline())
    logger.info('Important BUSINESS Features loaded')
    userImportantFeatures = json.loads(open(u_file,'r').readline())
    logger.info('Important USER Features loaded')
    trainReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        trainReviews.append(json.loads(line.strip()))
    
    #run function
    modelDict = learnFeatureExistance(busImportantFeatures, userImportantFeatures, trainReviews, path)
    
    #save model
    model_path = path+'models/'
    try:
        os.stat(model_path)
    except:
        os.mkdir(model_path)
    pickle.dump(modelDict,open(model_path+'modelDict_%d.model'%counter,'wb'))
    
    #save model results
    model_path = path+'models/'
    try:
        os.stat(model_path)
    except:
        os.mkdir(model_path)
    outfile = open(model_path+'modelStat_%d.txt'%counter,'w')
    feats = list(modelDict.keys())
    feats.sort()
    outfile.write('Feature_name\tStatistics\tExistence_F1\tSentiment_F1\n')
    for feat in feats:
        feat_info = modelDict[feat][4]
        str_feat_info = ['%5s'%str(x) for x in feat_info]
        outfile.write('%20s\t%s\n'%(feat,'\t'.join(str_feat_info)))
    outfile.close()
    
    return counter
#    model_path = path+'/modelPictures/'
#    for feature in modelDict:
#        dot_data = StringIO() 
#        tree.export_graphviz(modelDict[feature], out_file=dot_data) 
#        graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#        graph.write_pdf(model_path + feature + '.pdf')
    