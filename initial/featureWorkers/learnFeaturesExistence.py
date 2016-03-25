import json
import logging
import pickle
import random
import os
#import graphlab as gl
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import precision_recall_curve
#from sklearn.externals.six import StringIO
#from sklearn.cross_validation import KFold
from sklearn import cross_validation
#from sklearn import svm
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.decomposition import PCA


from utils.featuresStructure import featureStructureWorker
from getFeatures import getFeatures

#def getBestModel(logger, X, Y):
#    weights = [{0:1, 1:x} for x in np.logspace(-1.1, 1.1, 20)]
#    weights.append('auto')
#    lr = linear_model.LogisticRegression(C=1e5)
#    clf = GridSearchCV(estimator=lr, param_grid=dict(class_weight=weights), n_jobs=-1, scoring='f1')
#    clf = clf.fit(X, Y)
#    logger.info('f1: %.4f with %s'%(clf.best_score_,str(clf.best_estimator_.class_weight)))
#    return clf.best_score_, linear_model.LogisticRegression(C=1e5, class_weight=clf.best_estimator_.class_weight).fit(X, Y)

def getBestLogModel(logger, feature, X, Y, path):
    threses = list()
    qualities = list()
    
    CF1 = 0.0
    bestC = 0.5
    for C in [0.00001,0.0001,0.001,0.01,0.1,1]:
        indicator = range(len(X))
        random.shuffle(indicator)
        thres = int(len(indicator)*0.8)
        trainX = np.array([X[i] for i in indicator[:thres]])
        trainY = np.array([Y[i] for i in indicator[:thres]])
        testX = np.array([X[i] for i in indicator[thres:]])
        testY = np.array([Y[i] for i in indicator[thres:]])
        logmodel = linear_model.LogisticRegression(C=C, class_weight='auto').fit(trainX, trainY)
        Ypred = [x[1] for x in logmodel.predict_proba(testX)]
        prec, rec, thresholds = precision_recall_curve(testY, Ypred)
        bestF1 = 0.0
        bestThres = 0.0
        for i, thres in enumerate(thresholds[:-1]):
            #print thres, prec[i],rec[i]
            try:
                F1 = 2*prec[i]*rec[i]/(prec[i]+rec[i])
            except:
                continue
            #print F1, prec[i], rec[i]
            if F1 > bestF1:
                bestF1 = F1
                bestThres = thres
        
        print C, bestF1
        if bestF1 > CF1:
            bestC = C
            CF1 = bestF1
    
    logger.info('%s: best C = %f'%(feature,bestC))
    
    for counter in range(3):
        indicator = range(len(X))
        random.shuffle(indicator)
        thres = int(len(indicator)*0.8)
        trainX = np.array([X[i] for i in indicator[:thres]])
        trainY = np.array([Y[i] for i in indicator[:thres]])
        testX = np.array([X[i] for i in indicator[thres:]])
        testY = np.array([Y[i] for i in indicator[thres:]])
    
    
        
        logmodel = linear_model.LogisticRegression(C=bestC, class_weight='auto').fit(trainX, trainY)
        Ypred = [x[1] for x in logmodel.predict_proba(testX)]
        
        prec, rec, thresholds = precision_recall_curve(testY, Ypred)
        bestF1 = 0.0
        bestThres = 0.0
        
        
        for i, thres in enumerate(thresholds[:-1]):
            #print thres, prec[i],rec[i]
            try:
                F1 = 2*prec[i]*rec[i]/(prec[i]+rec[i])
            except:
                continue
            #print F1, prec[i], rec[i]
            if F1 > bestF1:
                bestF1 = F1
                bestThres = thres
        #print '=============='
        #print counter, bestF1, bestThres
        logger.info('Best score on LR with thres = %.3f is equal: %.3f'%(bestThres,bestF1))
        drawPR(feature,testY,Ypred,bestThres,bestF1, path, 'LR_%d'%counter)
        threses.append(bestThres)
        qualities.append(bestF1)
        
    X = np.array(X)
    Y = np.array(Y)
    logmodel = linear_model.LogisticRegression(C=1e5, class_weight='auto').fit(X, Y)
    bestThres = np.average(threses)
    bestF1 = np.average(qualities)
    #get quality on test!!!!!
    return bestThres,bestF1,logmodel




def getBestSVMModel(logger, feature, X, Y, path):
    threses = list()
    qualities = list()
    for counter in range(3):
        indicator = range(len(X))
        random.shuffle(indicator)
        thres = int(len(indicator)*0.8)
        trainX = np.array([X[i] for i in indicator[:thres]])
        trainY = np.array([Y[i] for i in indicator[:thres]])
        testX = np.array([X[i] for i in indicator[thres:]])
        testY = np.array([Y[i] for i in indicator[thres:]])
    
        svmmodel = SVC().fit(trainX, trainY)
        #Ypred = [x[1] for x in svmmodel.predict_proba(testX)]
        Ypred = svmmodel.predict(testX)
        
        prec, rec, thresholds = precision_recall_curve(testY, Ypred)
        bestF1 = 0.0
        bestThres = 0.0
        
        
        for i, thres in enumerate(thresholds[:-1]):
            #print thres, prec[i],rec[i]
            try:
                F1 = 2*prec[i]*rec[i]/(prec[i]+rec[i])
            except:
                continue
            #print F1, prec[i], rec[i]
            if F1 > bestF1:
                bestF1 = F1
                bestThres = thres
        #print '=============='
        #print counter, bestF1, bestThres
        logger.info('Best score on SVM with thres = %.3f is equal: %.3f'%(bestThres,bestF1))
        drawPR(feature,testY,Ypred,bestThres,bestF1, path, 'SVM_%d'%counter)
        threses.append(bestThres)
        qualities.append(bestF1)
        
    X = np.array(X)
    Y = np.array(Y)
    svmmodel = SVC().fit(X, Y)
    bestThres = np.average(threses)
    bestF1 = np.average(qualities)
    #get quality on test!!!!!
    return bestThres,bestF1,svmmodel





def crossValidation(logger, X, Y):
    models = dict()
    
    #models['DecisionTree'] = tree.DecisionTreeClassifier(criterion='entropy')
    models['LogisticRegression'] = linear_model.LogisticRegression(C=1e5, class_weight='auto')
    
#    lr = linear_model.LogisticRegression(C=1e5, class_weight='auto')
#    
##    weights = [{0:x} for x in np.logspace(-1, 1.1, 10)]
##    clf = GridSearchCV(estimator=lr, param_grid=dict(class_weight=weights), n_jobs=-1, scoring='recall')
##    clf = clf.fit(X, Y)
##    print 'anti_recall ',clf.best_score_, clf.best_estimator_.class_weight
#    
#    
#    weights = [{0:1, 1:x} for x in np.logspace(-1.1, 1.1, 20)]
#    weights.append('auto')
#    
##    clf = GridSearchCV(estimator=lr, param_grid=dict(class_weight=weights), n_jobs=-1, scoring='recall')
##    clf = clf.fit(X, Y)
##    print 'recall ',clf.best_score_, clf.best_estimator_.class_weight
##    
##    clf = GridSearchCV(estimator=lr, param_grid=dict(class_weight=weights), n_jobs=-1, scoring='precision')
##    clf = clf.fit(X, Y)
##    print 'precision ',clf.best_score_, clf.best_estimator_.class_weight
#    clf = GridSearchCV(estimator=lr, param_grid=dict(class_weight=weights), n_jobs=-1, scoring='f1')
#    clf = clf.fit(X, Y)
#    print 'f1 ',clf.best_score_, clf.best_estimator_.class_weight
    
    
    
    models['BernoulliNB'] = BernoulliNB()
    
    models['GaussianNB'] = GaussianNB()
    
#    models['GradientBoostingClassifier'] = ensemble.GradientBoostingClassifier(n_estimators=100,
#                                                                               learning_rate=1.0,
#                                                                               max_depth=3,
#                                                                               random_state=0)

    #models['RandomForestClassifier'] = ensemble.RandomForestClassifier(n_estimators=10, criterion='entropy', n_jobs=-1)
    #models['ExtraTreesClassifier'] = ensemble.ExtraTreesClassifier(n_estimators=10, criterion='entropy', n_jobs=-1)           
    #models['SGDClassifier'] = SGDClassifier(loss="hinge", penalty="l2")
    models['SVMrbf'] = SVC(kernel='rbf',class_weight='auto')
    models['SVMlinear'] = SVC(kernel='linear',class_weight='auto')
    models['SVMpoly'] = SVC(kernel='poly',class_weight='auto')
    #models['SVMsigmoid'] = SVC(kernel='sigmoid',class_weight='auto')

    #print 'Y',Y
    #print 'HELLO!!!'
    for model in models:
        #print model
        clf = models[model]
#        scores_p = cross_validation.cross_val_score(clf, X, Y, cv=10, scoring='precision')
#        scores_r = cross_validation.cross_val_score(clf, X, Y, cv=10, scoring='recall')
        scores_f1 = cross_validation.cross_val_score(clf, X, Y, cv=3, scoring='f1')
#        logger.info('%22s: Precision: %0.2f (+/- %0.2f) recall: %0.2f (+/- %0.2f)'%(model,scores_p.mean(),
#                                                                                        scores_p.std() * 2,
#                                                                                        scores_r.mean(),
#                                                                                        scores_r.std() * 2))
        logger.info('%22s: F1: %0.2f (+/- %0.2f)'%(model,
                                                                                    scores_f1.mean(),
                                                                                    scores_f1.std() * 2))
    
#    clf = svm.SVC()
##    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
##        gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
##        random_state=None, shrinking=True, tol=0.001, verbose=False)

#clf = AdaBoostClassifier(n_estimators=100)
    


        

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
    try:
        os.stat(path+'modelPictures/')
    except:
        os.mkdir(path+'modelPictures/')
    plt.savefig(path+'modelPictures/%s_pr_%s.png'%(feature,name))
    
    

def learnFeatureExistance(busImportantFeatures, userImportantFeatures, trainReviews, path):
    logger = logging.getLogger('signature.lFE.learnFE')
    logger.info('starting learnFeatureExistance from %d reviews'%len(trainReviews))
    fsw = featureStructureWorker()
    modelDict = dict()
    trainAveragesDict = dict()
    
    
    for f, feature in enumerate(fsw.featureIdicator):
        if not fsw.featureIdicator[feature]:
            continue
        logger.debug('Start working with %s'%feature)
        
        #get data
        X, Y, trainAveragesDict[feature] = getFeatures(logger, feature, trainReviews, busImportantFeatures, userImportantFeatures,
                                          trainAverages = {}, is_train = True)
        logger.debug('Got features for %d reviews'%len(X))
#        #cross validation
#        indicator = range(len(X))
#        random.shuffle(indicator)
#        thres = int(len(indicator)*0.8)
#        trainX = np.array([X[i] for i in indicator[:thres]])
#        trainY = np.array([Y[i] for i in indicator[:thres]])
#        testX = np.array([X[i] for i in indicator[thres:]])
#        testY = np.array([Y[i] for i in indicator[thres:]])
        
        #Logistic Regression
        #bestThres,bestF1,logmodel = getBestLogModel(logger, feature, trainX, trainY, testX, testY, X, Y, path)
        bestThres,bestF1,logmodel = getBestLogModel(logger, feature, X, Y, path)
        #bestThresSVM,bestF1SVM,svmmodel = getBestSVMModel(logger, feature, X, Y, path)
        
#       crossValidation(logger, np.array(X), np.array(Y))
        
        
        modelDict[feature] = [bestThres,bestF1,logmodel]
        
#        print f
#        if f > 6:
#            break
        
    return trainAveragesDict, modelDict


def learnFE(path, limit = np.Inf):
    logger = logging.getLogger('signature.lFE')
    logger.info('starting learnFE')
    #get data
    b_file = path+'/businessFeaturesAggregation_stat.json'
    u_file = path+'/userFeaturesAggregation_stat.json'
    r_file = path+'/yelp_reviews_features_extrain.json'
    
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
    trainAveragesDict, modelDict = learnFeatureExistance(busImportantFeatures, userImportantFeatures, trainReviews, path)
    
    #save model
    model_path = path+'models/'
    try:
        os.stat(model_path)
    except:
        os.mkdir(model_path)
    pickle.dump(modelDict,open(model_path+'modelDict_%d.model'%counter,'wb'))
    
    #save averages
    output = open(model_path+'trainAverages_%d.model'%counter,'wb')
    output.write(json.dumps(trainAveragesDict).encode('utf8', 'ignore'))
    output.close()
    
    return counter
#    model_path = path+'/modelPictures/'
#    for feature in modelDict:
#        dot_data = StringIO() 
#        tree.export_graphviz(modelDict[feature], out_file=dot_data) 
#        graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#        graph.write_pdf(model_path + feature + '.pdf')
    