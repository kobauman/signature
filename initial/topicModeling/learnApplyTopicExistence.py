import json
import logging
import pickle
import random
import os
import sys
sys.path.append('../')

import numpy as np
from sklearn.metrics import precision_recall_curve
#from sklearn.externals.six import StringIO
#from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

from sklearn.linear_model import Ridge

from utils.featuresStructure import featureStructureWorker
from getFeatures import getFeatures

def getBestModel(logger, topic, learnData, learnLabels, path):
    learnData = np.array(learnData)
    learnLabels = np.array(learnLabels)
    models = dict()
    
#    #models['DecisionTree'] = DecisionTreeRegressor(random_state=0, splitter = 'random', max_features = 'auto', min_samples_leaf = 50)
#    
    for alpha in np.logspace(-2, 0.01, 10):
        models['Ridge_%f'%alpha] = Ridge(alpha=alpha,normalize=True)
    
    
    models['Ridge_0.05'] = Ridge(alpha=0.05,normalize=True)  
#    models['SVMlinear'] = svm.SVR(kernel='linear', C=1e3)
#    models['SVMrbf'] = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
#    models['SVMpoly'] = svm.SVR(kernel='poly', C=1e3, degree=2)
    
    for model in models:
        print model
        clf = models[model]
        kf = cross_validation.KFold(len(learnLabels), n_folds=10)
        res = [[],[]]
        for train_index, test_index in kf:
            X_train, X_test = learnData[train_index], learnData[test_index]
            y_train, y_test = learnLabels[train_index], learnLabels[test_index]
            #print y_train
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            #print y_test,y_pred
            res[0].append(mean_squared_error(y_test, y_pred))
            res[1].append(explained_variance_score(y_test, y_pred))
            #logger.info('%s: \tMSE =  %f\tEV = %f'%(model,mean_squared_error(y_test, y_pred), explained_variance_score(y_test, y_pred)))
        print model
        print np.average(res[0]), np.average(res[1])

    models['Ridge_0.05'].fit(learnData,learnLabels)
    
    return models['Ridge_0.05']





    
    
def learnTopicExistence(busImportantFeatures, userImportantFeatures, trainReviews, path):
    logger = logging.getLogger('signature.lTE.learnTopicExistence')
    logger.info('starting learnTopicExistence from %d reviews'%len(trainReviews))
    fsw = featureStructureWorker()
    modelDict = dict()
    
    for f, topic in enumerate(fsw.featureIdicator):
        if not fsw.featureIdicator[topic]:
            continue
        logger.debug('Start working with %s'%topic)
        
        #get data
        X, Y = getFeatures(logger, topic, trainReviews, busImportantFeatures, userImportantFeatures,
                                          trainAverages = {}, is_train = True)
        logger.debug('Got features for %d reviews'%len(X))
        
        
        modelDict[topic] = getBestModel(logger, topic, X, Y, path)
    
    #print modelDict
    return modelDict


def learnTE(path, limit = np.Inf):
    logger = logging.getLogger('signature.lFE')
    logger.info('starting learnTE')
    #get data
    b_file = path+'businessFeaturesAggregation_stat.json'
    u_file = path+'userFeaturesAggregation_stat.json'
    r_file = path+'yelp_reviews_features_extrain.json'
    
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
    modelDict = learnTopicExistence(busImportantFeatures, userImportantFeatures, trainReviews, path)
    
    #save model
    model_path = path+'models/'
    try:
        os.stat(model_path)
    except:
        os.mkdir(model_path)
    pickle.dump(modelDict,open(model_path+'modelTopicDict_%d.model'%counter,'wb'))
    
    return counter



if __name__ == '__main__':
    logger = logging.getLogger('signature')

    logging.basicConfig(format='%(asctime)s : %(name)-12s: %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))
    
    #path  = '../../data/'
    #path = '../../data/bing/American_New'
    #path = '../../data/restaurants_topics_15/'
    path = '../../data/restaurants_topics_sent/'
    learnTE(path,1000000000)