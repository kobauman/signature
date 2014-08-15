
import numpy as np
from utils.featuresStructure import featureStructureWorker


def getBasicFeatures(feature, ID, dictionary):
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
        print ID
        return [None,None,None,None,None,None,None] + [None,None,None,None,None]
    
    
def getFeatures(logger, feature, reviewsSet, busImportantFeatures, userImportantFeatures,
                trainAverages = {}, train = True):
    if train:
        trainAverages = {'mean':[],'std':[]}
    else:
        pass
        #load trainAverages
    
    fsw = featureStructureWorker()
    X = list()
    Y = list()
    
    for review in reviewsSet:
        reviewFeatures = fsw.getReviewFeaturesExistence(review['features'])
        if feature in reviewFeatures:
            existance = 1
        else:
            existance = 0
            
        busID = review['business_id']
        userID = review['user_id']
            
        bus_basic_features = getBasicFeatures(feature, busID, busImportantFeatures)
        user_basic_features = getBasicFeatures(feature, userID, userImportantFeatures)
        bus_additional_features = []#getBusFeatures()
        user_additional_features = []#getUserFeatures()
        #sex = [review['usersSex']]
            
        Y.append(existance)
        X.append(bus_basic_features + user_basic_features + bus_additional_features + user_additional_features) #+sex)
        
        if train:
            if not len(trainAverages['mean']):
                for i in range(len(X[0])):
                    trainAverages['mean'].append([])
                    trainAverages['std'].append([])  
            
            for i, value in enumerate(X[-1]):
                if value!= None:
                    trainAverages['mean'][i].append(value)
    #count means               
    if train:        
        for i in range(len(trainAverages['mean'])):
            trainAverages['std'][i] = np.std(trainAverages['mean'][i])
            trainAverages['mean'][i] = np.average(trainAverages['mean'][i])
        
    #normalization   
    for vector in X:
        for i in range(len(vector)):
            if vector[i] == None:
                vector[i] = 0.0
            else:
                vector[i] = (vector[i]-trainAverages['mean'][i]) / trainAverages['std'][i]
    
    
    
    return X,Y,trainAverages