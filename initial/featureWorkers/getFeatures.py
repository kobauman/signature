import json
import numpy as np
from utils.featuresStructure import featureStructureWorker
from utils.sexPredictor import genderPredictor
from utils.categoryWorker import categoryWorker

def loadData(logger):
    train_data_path = "../../../yelp/data/splitting_reviews/"
    #load business information
    business_file = open(train_data_path + "yelp_training_set_business.json","r")
    business_dict = dict()
    for line in business_file:
        l = json.loads(line)
        business_dict[l['business_id']] = l
    logger.info('Loaded %d businesses'%len(business_dict))
        
    #load user information
    user_file = open(train_data_path + "yelp_training_set_user.json","r")
    user_dict = dict()
    for line in user_file:
        l = json.loads(line)
        user_dict[l['user_id']] = l
    logger.info('Loaded %d users'%len(user_dict))
    return business_dict, user_dict


def getBasicFeatures(feature, ID, dictionary, train = True):
    if ID in dictionary:
        tfidf = dictionary[ID]['tfidfDict'].get(feature,0.0)
        freq = int(dictionary[ID]['featureFreq'].get(feature,0.0)*dictionary[ID]['reviewsNumber']/100)
        logfreq = np.log(1 + freq)
        pfreq = dictionary[ID]['featureFreq'].get(feature,0.0)
        sent = dictionary[ID]['sentiment'].get(feature,[0.0,0])[0]
        reviewNum = dictionary[ID]['reviewsNumber']
        logreviewNum = np.log(1 + reviewNum)
        maxFreq = dictionary[ID]['maxFreq']
        featureNum = len(dictionary[ID]['tfidfDict'])
        importantFeatureNum = len([f for f in dictionary[ID]['featureFreq'] if dictionary[ID]['featureFreq'][f] > 10 and
                                   dictionary[ID]['sentiment'][f][1] > 1])
        textFeatures = dictionary[ID]['textFeatures']
        result = [tfidf,logfreq,pfreq,sent,reviewNum,logreviewNum ,maxFreq,featureNum,importantFeatureNum] + textFeatures
    else:
        #print ID
        result = [None,None,None,None,None,None,None,None,None] + [None,None,None,None,None]
    
#    if not train or (result[4] and result[4] > 10):
#        return result
#    else:
#        return False
    return result
        
    



def getBusinessFeatures(busID, business_dict, catWorker):
    if busID not in business_dict:
        return [None]*catWorker.vec_len
    cats = business_dict[busID]['categories']
    vector = catWorker.classify(cats)
    #print cats,vector
    return vector

def getUserFeatures(userID, user_dict, genderPredictor):
    if userID not in user_dict:
        return [None]
    name = user_dict[userID]['name']
    gender = genderPredictor.classify(name)
    return [gender]
    
def getFeatures(logger, feature, reviewsSet, busImportantFeatures, userImportantFeatures,
                trainAverages = {}, is_train = True):
    
    business_dict, user_dict = loadData(logger)
    gP = genderPredictor()
    gP.load()
    cW = categoryWorker()
    cW.load()
    
    if is_train:
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
            
        bus_basic_features = getBasicFeatures(feature, busID, busImportantFeatures, is_train)
        user_basic_features = getBasicFeatures(feature, userID, userImportantFeatures, is_train)
        bus_additional_features = getBusinessFeatures(busID, business_dict, cW)
        user_additional_features = getUserFeatures(userID, user_dict, gP)
        
#        if not bus_basic_features or not user_basic_features:
#            continue
        #sex = [review['usersSex']]
            
        Y.append(existance)
        X.append(bus_basic_features + user_basic_features + bus_additional_features + user_additional_features) #+sex)
        
        if is_train:
            if not len(trainAverages['mean']):
                for i in range(len(X[0])):
                    trainAverages['mean'].append([])
                    trainAverages['std'].append([])  
            
            for i, value in enumerate(X[-1]):
                if value!= None:
                    trainAverages['mean'][i].append(value)
    #count means               
    if is_train:        
        for i in range(len(trainAverages['mean'])):
            trainAverages['std'][i] = np.std(trainAverages['mean'][i])
            trainAverages['mean'][i] = np.average(trainAverages['mean'][i])
        
    #normalization   
    for vector in X:
        for i in range(len(vector)):
            if vector[i] == None:
                vector[i] = 0.0
            else:
                if trainAverages['std'][i]:
                    vector[i] = (vector[i]-trainAverages['mean'][i]) / trainAverages['std'][i]
                else:
                    vector[i] = (vector[i]-trainAverages['mean'][i])
    
    
    if is_train:
        return X,Y,trainAverages
    else:
        return X,Y