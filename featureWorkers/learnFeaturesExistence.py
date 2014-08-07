import json
import pydot 
import logging
import pickle

import numpy as np
from sklearn import tree
from sklearn.externals.six import StringIO
#from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn import svm

from utils.featuresStructure import featureStructureWorker

def crossValidation(logger, X, Y):

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    scores_p = cross_validation.cross_val_score(clf, X, Y, cv=10, scoring='precision')
    scores_r = cross_validation.cross_val_score(clf, X, Y, cv=10, scoring='recall')
    logger.info('DecisionTree: Precision: %0.2f (+/- %0.2f) recall: %0.2f (+/- %0.2f)'%(scores_p.mean(),
                                                                                        scores_p.std() * 2,
                                                                                        scores_r.mean(),
                                                                                        scores_r.std() * 2))
    
    clf = svm.SVC()
    scores_p = cross_validation.cross_val_score(clf, X, Y, cv=10, scoring='precision')
    scores_r = cross_validation.cross_val_score(clf, X, Y, cv=10, scoring='recall')
    logger.info('SVM: Precision: %0.2f (+/- %0.2f) recall: %0.2f (+/- %0.2f)'%(scores_p.mean(),
                                                                                        scores_p.std() * 2,
                                                                                        scores_r.mean(),
                                                                                        scores_r.std() * 2))

    

def learnFeatureExistance(busImportantFeatures, userImportantFeatures, trainReviews):
    logger = logging.getLogger('signature.lFE.learnFE')
    logger.info('starting learnFeatureExistance from %d reviews'%len(trainReviews))
    fsw = featureStructureWorker()
    modelDict = dict()
    for i, feature in enumerate(fsw.featureIdicator):
        logger.debug('Start working with %s'%feature)
        X = list()
        Y = list()
        if not fsw.featureIdicator[feature]:
            continue
        for review in trainReviews:
            reviewFeatures = fsw.getReviewFeatures(review['features'])
            if feature in reviewFeatures:
                existance = 1
            else:
                existance = 0
            
            busID = review['business_id']
            userID = review['user_id']
            
            if busID in busImportantFeatures:
                tfidfB = busImportantFeatures[busID]['tfidfDict'].get(feature,0.0)
                freqB = int(busImportantFeatures[busID]['featureFreq'].get(feature,0.0)*busImportantFeatures[busID]['reviewsNumber']/100)
                pfreqB = busImportantFeatures[busID]['featureFreq'].get(feature,0.0)
            else:
                tfidfB,freqB,pfreqB = 0.0, 0.0, 0.0
            
            if userID in userImportantFeatures:
                tfidfU = userImportantFeatures[userID]['tfidfDict'].get(feature,0.0)
                freqU = int(userImportantFeatures[userID]['featureFreq'].get(feature,0.0)*userImportantFeatures[userID]['reviewsNumber']/100)
                pfreqU = userImportantFeatures[userID]['featureFreq'].get(feature,0.0)
            else:
                tfidfU,freqU,pfreqU = 0.0,0.0,0.0
            X.append([tfidfB,freqB,pfreqB,tfidfU,freqU,pfreqU])
            Y.append(existance)
            #print existance, [tfidfB,freqB,pfreqB,tfidfU,freqU,pfreqU]
        
        
#        kf = KFold(len(X), n_folds=5)
#        eval = {'entropy':[],'gini':[]}
#        for train, test in kf:
#            X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
#            clf_entropy = tree.DecisionTreeClassifier(max_depth=4,criterion='entropy')
        
        X = np.array(X)
        Y = np.array(Y)   
        crossValidation(logger, X, Y)
        
        
            
        modelDict[feature] = tree.DecisionTreeClassifier(max_depth=4,criterion='entropy')
        modelDict[feature] = modelDict[feature].fit(X, Y)
        logger.info('Score on train: %s'%str(modelDict[feature].score(X,Y)))
        if i > 4:
            break
        #break
    return modelDict


def learnFE(path):
    logger = logging.getLogger('signature.lFE')
    logger.info('starting learnFE')
    #get data
    b_file = path+'/businessFeaturesAggregation.json'
    u_file = path+'/userFeaturesAggregation.json'
    r_file = path+'/yelp_reviews_features_train.json'
    
    busImportantFeatures = json.loads(open(b_file,'r').readline())
    userImportantFeatures = json.loads(open(u_file,'r').readline())
    logger.info('ImportantFeatures loaded')
    trainReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > 25000:
            break
        trainReviews.append(json.loads(line.strip()))
    
    #run function
    modelDict = learnFeatureExistance(busImportantFeatures, userImportantFeatures, trainReviews)
    
    #save model
    #pickle.dumps(modelDict, )
    
    model_path = path+'/modelPictures/'
    for feature in modelDict:
        dot_data = StringIO() 
        tree.export_graphviz(modelDict[feature], out_file=dot_data) 
        graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
        graph.write_pdf(model_path + feature + '.pdf')
    