import json
import logging
import pickle
import random
#import graphlab as gl
import matplotlib.pyplot as plt

import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
#from sklearn.externals.six import StringIO
#from sklearn.cross_validation import KFold
#from sklearn import cross_validation
#from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.decomposition import PCA

from utils.featuresStructure import featureStructureWorker


def getBestModel(logger, X, Y):
    weights = [{0:1, 1:x} for x in np.logspace(-1.1, 1.1, 20)]
    weights.append('auto')
    lr = linear_model.LogisticRegression(C=1e5)
    clf = GridSearchCV(estimator=lr, param_grid=dict(class_weight=weights), n_jobs=-1, scoring='f1')
    clf = clf.fit(X, Y)
    logger.info('f1: %.4f with %s'%(clf.best_score_,str(clf.best_estimator_.class_weight)))
    return clf.best_score_, linear_model.LogisticRegression(C=1e5, class_weight=clf.best_estimator_.class_weight).fit(X, Y)

def crossValidation(logger, X, Y):
    #!!!PCA
    
    models = dict()
    
    models['DecisionTree'] = tree.DecisionTreeClassifier(criterion='entropy')
    #models['SVM'] = svm.SVC()
    models['LogisticRegression'] = linear_model.LogisticRegression(C=1e5, class_weight='auto')
    
    lr = linear_model.LogisticRegression(C=1e5, class_weight='auto')
    
#    weights = [{0:x} for x in np.logspace(-1, 1.1, 10)]
#    clf = GridSearchCV(estimator=lr, param_grid=dict(class_weight=weights), n_jobs=-1, scoring='recall')
#    clf = clf.fit(X, Y)
#    print 'anti_recall ',clf.best_score_, clf.best_estimator_.class_weight
    
    
    weights = [{0:1, 1:x} for x in np.logspace(-1.1, 1.1, 20)]
    weights.append('auto')
    
#    clf = GridSearchCV(estimator=lr, param_grid=dict(class_weight=weights), n_jobs=-1, scoring='recall')
#    clf = clf.fit(X, Y)
#    print 'recall ',clf.best_score_, clf.best_estimator_.class_weight
#    
#    clf = GridSearchCV(estimator=lr, param_grid=dict(class_weight=weights), n_jobs=-1, scoring='precision')
#    clf = clf.fit(X, Y)
#    print 'precision ',clf.best_score_, clf.best_estimator_.class_weight
    clf = GridSearchCV(estimator=lr, param_grid=dict(class_weight=weights), n_jobs=-1, scoring='f1')
    clf = clf.fit(X, Y)
    print 'f1 ',clf.best_score_, clf.best_estimator_.class_weight
    
    
    
    
    models['GaussianNB'] = GaussianNB()
#    models['GradientBoostingClassifier'] = ensemble.GradientBoostingClassifier(n_estimators=100,
#                                                                               learning_rate=1.0,
#                                                                               max_depth=3,
#                                                                               random_state=0)

    models['RandomForestClassifier'] = ensemble.RandomForestClassifier(n_estimators=10, criterion='entropy', n_jobs=-1)
    models['ExtraTreesClassifier'] = ensemble.ExtraTreesClassifier(n_estimators=10, criterion='entropy', n_jobs=-1)           
    models['SGDClassifier'] = SGDClassifier(loss="hinge", penalty="l2")


#    for model in models:
#        clf = models[model]
#        scores_p = cross_validation.cross_val_score(clf, X, Y, cv=10, scoring='precision')
#        scores_r = cross_validation.cross_val_score(clf, X, Y, cv=10, scoring='recall')
#        logger.info('%22s: Precision: %0.2f (+/- %0.2f) recall: %0.2f (+/- %0.2f)'%(model,scores_p.mean(),
#                                                                                        scores_p.std() * 2,
#                                                                                        scores_r.mean(),
#                                                                                        scores_r.std() * 2))
    
#    clf = svm.SVC()
##    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
##        gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
##        random_state=None, shrinking=True, tol=0.001, verbose=False)

#clf = AdaBoostClassifier(n_estimators=100)
    

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
        

def drawPR(feature,y_true,y_pred,Yt,Ypred,path):
    precision_mf, recall_mf, thresholds_mf = precision_recall_curve(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(Yt, Ypred)
    
    
    # Create plots with pre-defined labels.
    # Alternatively, you can pass labels explicitly when calling `legend`.
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.set_title('%s first prediction'%feature)
    ax0.plot(thresholds_mf, precision_mf[:-1], 'k--', color = 'green', label='precision_mf')
    ax0.plot(thresholds_mf, recall_mf[:-1], 'k:', color = 'green', label='recall_mf')
    ax0.legend(shadow=True)
    
    
    ax1.set_title('%s second prediction'%feature)
    ax1.plot(thresholds, precision[:-1], 'k--', color = 'red', label='precision')
    ax1.plot(thresholds, recall[:-1], 'k:', color = 'red', label='recall')
    ax1.legend(shadow=True)
    
    #plt.show()
    plt.savefig(path+'/modelPictures/%s_pr.png'%feature)
    
    

def learnFeatureExistance(busImportantFeatures, userImportantFeatures, trainReviews, path):
    logger = logging.getLogger('signature.lFE.learnFE')
    logger.info('starting learnFeatureExistance from %d reviews'%len(trainReviews))
    fsw = featureStructureWorker()
    modelDict = dict()
    for i, feature in enumerate(fsw.featureIdicator):
#        if feature.count('_')>1:
#            continue
        logger.debug('Start working with %s'%feature)
        X = list()
        Y = list()
#        data = {'user':[],'item':[],'rating':[]}
        Xt = list()
        Yt = list()
#        datat = {'user':[],'item':[],'rating':[]}
        if not fsw.featureIdicator[feature]:
            continue
        for review in trainReviews:
            reviewFeatures = fsw.getReviewFeaturesExistence(review['features'])
            if feature in reviewFeatures:
                existance = 1
            else:
                existance = 0
            
            busID = review['business_id']
            userID = review['user_id']
            
            bus_features = getFeatures(feature, busID, busImportantFeatures)
            user_features = getFeatures(feature, userID, userImportantFeatures)
            
            if random.random() > 0.8:
                Xt.append(bus_features + user_features)
                Yt.append(existance)
                #print existance, [tfidfB,freqB,pfreqB,tfidfU,freqU,pfreqU]
                
#                datat['user'].append(userID)
#                datat['item'].append(busID)
#                datat['rating'].append(existance)
            else:
                X.append(bus_features + user_features)
                Y.append(existance)
                #print existance, [tfidfB,freqB,pfreqB,tfidfU,freqU,pfreqU]
                
#                data['user'].append(userID)
#                data['item'].append(busID)
#                data['rating'].append(existance)
 
        
        X = np.array(X)
        Y = np.array(Y)
        Xt = np.array(Xt)
        Yt = np.array(Yt)
#        train_set = gl.SFrame(data)
#        test_set = gl.SFrame(datat)
#        #ADD CROSSS VALIDATION
#        model = gl.recommender.create(train_set,user_column='user',item_column='item',
#                                                   target_column='rating',method='matrix_factorization',
#                                                   n_factors=7,regularization=100,
#                                                   binary_targets=True,
#                                                   max_iterations=50,verbose=False)
#        
#        y_true_mf = list(test_set['rating'])
#        y_pred_mf = list(model.score(test_set))

        logmodel = linear_model.LogisticRegression(C=1e5, class_weight='auto').fit(X, Y)
        Ypred = [x[1] for x in logmodel.predict_proba(Xt)]
#        print y_pred_mf[:50]
#        print y_true_mf[:50]
#        print Ypred[:50]
#        print Yt[:50]
        #drawPR(feature,y_true_mf,y_pred_mf,Yt,Ypred,path)
        
        
        quality, model = getBestModel(logger, X, Y)
        y_pred_mf = [x[1] for x in model.predict_proba(Xt)]
        y_true_mf = Yt
        logger.info('Score on best model: %.3f'%quality)
        drawPR(feature,y_true_mf,y_pred_mf,Yt,Ypred,path)
        modelDict[feature] = [quality, model]
        
        
#        modelDict[feature] = gl.recommender.create(learnData,user_column='user',item_column='item',
#                                                   target_column='rating',method='matrix_factorization',
#                                                   n_factors=7,regularization=100,
#                                                   #binary_targets=True,
#                                                   max_iterations=50,verbose=False)
        
        
        
        
        #crossValidation(logger, X, Y)
        
        #quality, model = getBestModel(logger, X, Y)
            
        #modelDict[feature] = getBestModel(logger, X, Y)
        #logger.info('Score on train: %s'%str(modelDict[feature].score(X,Y)))
#        if i > 15:
#            break
        #break
    return modelDict


def learnFE(path, limit = 100000000000000):
    logger = logging.getLogger('signature.lFE')
    logger.info('starting learnFE')
    #get data
    b_file = path+'/businessFeaturesAggregation_train.json'
    u_file = path+'/userFeaturesAggregation_train.json'
    r_file = path+'/yelp_reviews_features_train.json'
    
    busImportantFeatures = json.loads(open(b_file,'r').readline())
    userImportantFeatures = json.loads(open(u_file,'r').readline())
    logger.info('ImportantFeatures loaded')
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
    model_path = path+'/models/'
    pickle.dump(modelDict,open(model_path+'modelDict_%d.model'%counter,'wb'))
#    model_path = path+'/modelPictures/'
#    for feature in modelDict:
#        dot_data = StringIO() 
#        tree.export_graphviz(modelDict[feature], out_file=dot_data) 
#        graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#        graph.write_pdf(model_path + feature + '.pdf')
    