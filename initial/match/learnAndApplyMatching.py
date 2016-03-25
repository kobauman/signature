import json
import logging


import numpy as np

from utils.featuresStructure import featureStructureWorker
from match.getFeatures import getFeatures




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
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn import svm

from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
#explained_variance_score(y_true, y_pred)
#mean_squared_error(y_true, y_pred)
from sklearn import cross_validation


def learnMatchModel(logger, learnData, learnLabels):
    learnData = np.array(learnData)
    learnLabels = np.array(learnLabels)
    models = dict()
    
#    #models['DecisionTree'] = DecisionTreeRegressor(random_state=0, splitter = 'random', max_features = 'auto', min_samples_leaf = 50)
#    
#    for alpha in np.logspace(-2, 0.01, 10):
#        models['Ridge_%f'%alpha] = Ridge(alpha=alpha,normalize=True)
    
    
    models['Ridge_0.05'] = Ridge(alpha=0.05,normalize=True)  
#    models['SVMlinear'] = svm.SVR(kernel='linear', C=1e3)
#    models['SVMrbf'] = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
#    models['SVMpoly'] = svm.SVR(kernel='poly', C=1e3, degree=2)
    
#    for model in models:
#        print model
#        clf = models[model]
#        kf = cross_validation.KFold(len(learnLabels), n_folds=10)
#        res = [[],[]]
#        for train_index, test_index in kf:
#            X_train, X_test = learnData[train_index], learnData[test_index]
#            y_train, y_test = learnLabels[train_index], learnLabels[test_index]
#            clf.fit(X_train,y_train)
#            y_pred = clf.predict(X_test)
#            res[0].append(mean_squared_error(y_test, y_pred))
#            res[1].append(explained_variance_score(y_test, y_pred))
#            #logger.info('%s: \tMSE =  %f\tEV = %f'%(model,mean_squared_error(y_test, y_pred), explained_variance_score(y_test, y_pred)))
#        print model
#        print np.average(res[0]), np.average(res[1])

    models['Ridge_0.05'].fit(learnData,learnLabels)
    
    return models['Ridge_0.05']
        



def learnAndApplyMatching(path, limit = np.Inf):
    logger = logging.getLogger('signature.learnAndApplyMatching')
    logger.info('starting learnAndApplyMatching')
    #get data
    b_file = path+'businessFeaturesAggregation_stat.json'
    u_file = path+'userFeaturesAggregation_stat.json'
    train_file = path+'yelp_reviews_features_extrain.json'
    test_file = path+'yelp_reviews_test_predictions.json'
    
    busImportantFeatures = json.loads(open(b_file,'r').readline())
    logger.info('Important BUSINESS Features loaded')
    userImportantFeatures = json.loads(open(u_file,'r').readline())
    logger.info('Important USER Features loaded')
    fsw = featureStructureWorker()
    
#    #load featureWeights
#    infile = open(path+'/featureWeights.json','r')
#    featureWeights = json.loads(infile.readline().strip())
#    infile.close()
    
    #learn model
    learnData = list()
    learnLabels = list()
    for counter, line in enumerate(open(train_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        review = json.loads(line.strip())
        userID = review['user_id']
        busID = review['business_id']
        
        features = getFeatures(busID, userID, busImportantFeatures, userImportantFeatures, fsw)
        if features:
            learnData.append(features)
            learnLabels.append(review['stars'])
    
    model = learnMatchModel(logger, learnData, learnLabels)
    print model.coef_
    for aspect in fsw.featureIdicator:
        if not fsw.featureIdicator[aspect]:
            continue
        print aspect
#    print model
#    exit()
    
    #apply model
    testReviews = []
    for counter, line in enumerate(open(test_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        review = json.loads(line.strip())
        testReviews.append(review)
        
    
    outfile = open(path+'yelp_reviews_test_predictions.json','w')
    for counter, review in enumerate(testReviews):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        userID = review['user_id']
        busID = review['business_id']
        
        test_features = getFeatures(busID, userID, busImportantFeatures, userImportantFeatures, fsw)
        if test_features:
            prediction = model.predict(test_features)
        else:
            prediction = None
        
        review['rating_prediction'] = review.get('rating_prediction', {})
        review['rating_prediction']['match_prediction'] = prediction
        
        outfile.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
    outfile.close()
    
    