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
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA

from utils.featuresStructure import featureStructureWorker



def getFeatures(feature, ID, dictionary):
    if ID in dictionary:
        tfidf = dictionary[ID]['tfidfDict'].get(feature,0.0)
        freq = int(dictionary[ID]['featureFreq'].get(feature,0.0)*dictionary[ID]['reviewsNumber']/100)
        pfreq = dictionary[ID]['featureFreq'].get(feature,0.0)
        sent = dictionary[ID]['sentiment'].get(feature,[0.0,0])[0]
        reviewNum = dictionary[ID]['reviewsNumber']
        maxFreq = dictionary[ID]['maxFreq']
        featureNum = len(dictionary[ID]['tfidfDict'])
        textFeatures = dictionary[ID]['textFeatures']
        return [tfidf,freq,pfreq,sent,reviewNum,maxFreq,featureNum] + textFeatures
    else:
        return [0.0,0,0.0,0.0,0,0,0] + [0.0,0.0,0.0,0.0,0.0]
        


def applyFeatureExistance(busImportantFeatures, userImportantFeatures, testReviews, modelDict):
    logger = logging.getLogger('signature.aFE.applyFE')
    logger.info('starting applyFeatureExistance from %d reviews'%len(testReviews))
    fsw = featureStructureWorker()
    
    for r, review in enumerate(testReviews):
        reviewFeatures = fsw.getReviewFeaturesSentiment(review['features'])
        review['exPredFeatures'] = dict()
        for i, feature in enumerate(fsw.featureIdicator):
            if not fsw.featureIdicator[feature]:
                continue
            if feature in reviewFeatures:
                existance = 1
            else:
                existance = 0
                
            busID = review['business_id']
            userID = review['user_id']    
            bus_features = getFeatures(feature, busID, busImportantFeatures)
            user_features = getFeatures(feature, userID, userImportantFeatures)
            
            position = list(modelDict[feature][1].classes_).index(1)
            predictedExistance = int(modelDict[feature][1].predict_proba(bus_features + user_features)[0][position])
            
            #check if feature important
            if existance + predictedExistance >0.5:
                review['exPredFeatures'][feature] = [existance, predictedExistance]
        #print review['exPredFeatures']
        if not r%1000:
            logger.debug('%d reviews processed'%r)
#        X = np.array(X)
#        Y = np.array(Y)
        
    return testReviews


def applyFE(path, modelfile):
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
#        if counter > 20:
#            break
        testReviews.append(json.loads(line.strip()))
    logger.info('Reviews loaded')
    
    #load model
    modelDict = pickle.load(open(modelfile,'r'))
    logger.info('Model loaded from %s'%modelfile)
    
    #run function
    reviewsPrediction = applyFeatureExistance(busImportantFeatures, userImportantFeatures, testReviews, modelDict)
    
    #save result
    outfile = open(path+'/yelp_reviews_features_test_pF.json','w')
    for review in reviewsPrediction:
        outfile.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
    outfile.close()
    