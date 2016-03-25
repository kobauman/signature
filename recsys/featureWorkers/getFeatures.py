import json
import numpy as np
from scipy import stats
from utils.featuresStructure import featureStructureWorker
#from utils.sexPredictor import genderPredictor
#from utils.categoryWorker import categoryWorker

def sig_dif(array1,array2):
    try:
        arr1 = np.array(array1)
        arr2 = np.array(array2)
        return round(stats.ttest_ind(arr1,arr2)[1],3)
    except:
        return None

def compare(array1, array2):
    if int(sig_dif(array1,array2) < 0.10501):
        return [#int(np.average(array1) < np.average(array2)),
                np.average(array1) - np.average(array2),
                len(array1)/(len(array1) + len(array2)),
                #abs(np.average(array1) - np.average(array2))
                ]
    else:
        return [#0,
                0,
                0.5,
                #0
                ]

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


def getCriticalFeatures(feature, ID, dictionary):
    if ID in dictionary and feature in dictionary[ID]['critical']:
        exist = dictionary[ID]['critical'][feature]['1']
        pos = dictionary[ID]['critical'][feature]['+']
        neg = dictionary[ID]['critical'][feature]['-']
        neutr = dictionary[ID]['critical'][feature]['0']
        none = dictionary[ID]['critical'][feature]['n']
        
        result = []
        result += compare(exist,none)
        result += compare(pos,neg)
#        result += compare(pos,none)
#        result += compare(neg,none)
#        result += compare(neutr,none)
#        result += compare(pos+neutr,none)
#        result += compare(neg+neutr,none)
        return result
    else:
        return None
    
        


def calculateFeatures(logger, review, feature, busImportantFeatures, userImportantFeatures):        
    busID = review['business_id']
    userID = review['user_id']
    if busID not in busImportantFeatures or userID not in userImportantFeatures:
        return None
    
    bus_tfidf = busImportantFeatures[busID]['tfidfDict'].get(feature,0.0)
    bus_freq = busImportantFeatures[busID]['featureFreq'].get(feature,0.0)/100.0
    bus_reviews = busImportantFeatures[busID]['reviewsNumber']
    bus_sentiment = (busImportantFeatures[review['business_id']]['sentiment'].get(feature,[0.0,0])[0]+1)/2.0
    
    user_tfidf = userImportantFeatures[userID]['tfidfDict'].get(feature,0.0)
    user_freq = userImportantFeatures[userID]['featureFreq'].get(feature,0.0)/100.0
    user_reviews = userImportantFeatures[userID]['reviewsNumber']
    user_sentiment = (userImportantFeatures[review['user_id']]['sentiment'].get(feature,[0.0,0])[0]+1)/2.0
    user_text = userImportantFeatures[userID]['textFeatures']
    
    '''CHECK IF WE HAVE ENOUGH INFORMATION'''
    if bus_reviews > 5 and bus_freq > 0.1 and user_reviews > 5: # 5 0.1 5 # 0 0.1 0
        feature_set = [bus_tfidf,  bus_freq, bus_sentiment,
                       user_tfidf, user_freq, user_sentiment]
#            feature_set = [bus_freq]
        
        feature_set += getCriticalFeatures(feature, busID, busImportantFeatures)
        feature_set += getCriticalFeatures(feature, userID, userImportantFeatures)

        return feature_set
    
    else:
        return None


    
def getFeatures(logger, feature, reviewsSet, busImportantFeatures, userImportantFeatures):
    
    #business_dict, user_dict = loadData(logger)
    
    fsw = featureStructureWorker()
    X1 = list()
    Y1 = list()
    X2 = list()
    Y2 = list()
    
    missed = 0
    
    for review in reviewsSet:
        feature_set = calculateFeatures(logger, review, feature, busImportantFeatures, userImportantFeatures)
        
        
        reviewFeatures = fsw.getReviewFeaturesExistence(review['features'])
#            
#        busID = review['business_id']
#        userID = review['user_id']
#        if busID not in busImportantFeatures or userID not in userImportantFeatures:
#            missed += 1
#            continue
#        
#        bus_tfidf = busImportantFeatures[busID]['tfidfDict'].get(feature,0.0)
#        bus_freq = busImportantFeatures[busID]['featureFreq'].get(feature,0.0)/100.0
#        bus_reviews = busImportantFeatures[busID]['reviewsNumber']
#        bus_sentiment = (busImportantFeatures[review['business_id']]['sentiment'].get(feature,[0.0,0])[0]+1)/2.0
#        
#        user_tfidf = userImportantFeatures[userID]['tfidfDict'].get(feature,0.0)
#        user_freq = userImportantFeatures[userID]['featureFreq'].get(feature,0.0)/100.0
#        user_reviews = userImportantFeatures[userID]['reviewsNumber']
#        user_sentiment = (userImportantFeatures[review['user_id']]['sentiment'].get(feature,[0.0,0])[0]+1)/2.0
#        user_text = userImportantFeatures[userID]['textFeatures']
#        
#        '''CHECK IF WE HAVE ENOUGH INFORMATION'''
#        if bus_reviews > 5 and bus_freq > 0.1 and user_reviews > 5: # 5 1 5
#            feature_set = [bus_tfidf,  bus_freq, bus_sentiment,
#                           user_tfidf, user_freq, user_sentiment]
##            feature_set = [bus_freq]
#            
#            feature_set += getCriticalFeatures(feature, busID, busImportantFeatures)
#            feature_set += getCriticalFeatures(feature, userID, userImportantFeatures)
#            
##            feature_set += user_text
        if feature_set:
            
            if feature in reviewFeatures:
                Y1.append(1)
                X1.append(feature_set)
                
                sent = np.average(reviewFeatures[feature])
                if sent > 0:
                    Y2.append(1)
                    X2.append(feature_set)
                elif sent < 0:
                    Y2.append(0)
                    X2.append(feature_set)
            else:
                Y1.append(0)
                X1.append(feature_set)
        else:
            missed += 1
            
    return X1, Y1, X2, Y2, missed