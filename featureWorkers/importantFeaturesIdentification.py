import json
import re, glob
import sys
sys.path.append('../')
import logging

from utils.featuresStructure import featureStructureWorker
from utils.featureItemImportance import featureImportance
from utils.featureAggregation import featureAggregation

'''
7) Obtaining a set of important features (G_{r_i}) with weights for each Restaurant r_i
Parameters of importance for feature in reviews of particular restaurant:

   * number of reviews
   * % of reviews
   * TF-IDF technique
Methods: topN, by threshold, combination

Obtain two groups of features
a) High-frequent (HF_{r_i}) by threshold on % of reviews
b) Frequent (F_{r_i}) by other methods

Input:
   * set of interesting features
   * set of reviews (with features and sentiments) assigned to restaurants
Output:
restaurantID -> set of features G_{r_i} = (HF_{r_i} and F_{r_i})

'''

def importantFeatureIdentification(infileName, outBusinesFile, outUserFile, busInfo = False, userInfo = False):
    logger = logging.getLogger('signature.IFI')
    logger.info('starting importantFeatureIdentification from %s'%infileName)
    
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
    
    
    # dictionaries of reviews
    # key = businessID
    bus_reviews = dict()
    # key = userID
    user_reviews = dict()
    
    #load reviews
    review_file = open(infileName,"r")
    for counter, line in enumerate(review_file):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
#        if counter > 10000:
#            break
        # load review information
        review = json.loads(line)
        businessID = review['business_id']
        userID = review['user_id']
        
        # fill up dictionaries of reviews
        if businessID in business_dict:
            bus_reviews[businessID] = bus_reviews.get(businessID,[])
            bus_reviews[businessID].append(review['features'])
            
        if userID in user_dict:
            user_reviews[userID] = user_reviews.get(userID,[])
            user_reviews[userID].append(review['features'])    
    review_file.close()
    
    logger.info('Important feature identification for %d BUSINESSES'%len(bus_reviews))
    busImportantFeatures = featureImportance(bus_reviews)
    for bus in busImportantFeatures:
        busImportantFeatures[bus]['categories'] = business_dict[bus]['categories']
        busImportantFeatures[bus]['name'] = business_dict[bus]['name']
        busImportantFeatures[bus]['stars'] = business_dict[bus]['stars']
        
    if busInfo:
        out_file = open(outBusinesFile,"w")
        out_file.write(json.dumps(busImportantFeatures).encode('utf8', 'ignore'))
        out_file.close()
        
        out_file = open(outBusinesFile.replace('.json','.txt'),"w")
        for bus in busImportantFeatures:
            out_file.write('%s\t%s\nstars = %s\n'%(busImportantFeatures[bus]['name'].encode('utf8', 'ignore'),
                                                 str(busImportantFeatures[bus]['categories']).encode('utf8', 'ignore'),
                                                 str(busImportantFeatures[bus]['stars']).encode('utf8', 'ignore')))
            out_file.write('%20sFeatures\tTFIDF\tFreq\tSentiment\n'%'')
            fA = featureAggregation(bus_reviews[bus])
            featuresInfo = list()
            for feature in busImportantFeatures[bus]['tfidfDict']:
                if feature in busImportantFeatures[bus]['featureFreq'] and feature in fA:
                    featuresInfo.append([busImportantFeatures[bus]['tfidfDict'][feature],
                                         '%30s\t%s\t%s\t%s\n'%(feature,
                                                             str(busImportantFeatures[bus]['tfidfDict'][feature]),
                                                             str(busImportantFeatures[bus]['featureFreq'][feature]),
                                                             str(fA[feature]))])
            
            
            featuresInfo.sort(reverse=True)
            out_file.write(''.join([x[1] for x in featuresInfo]))           
            out_file.write('======================================================\n\n')                         
            
        out_file.close()
    
    
    
    logger.info('Important feature identification for %d USERS'%len(user_reviews))
    userImportantFeatures = featureImportance(user_reviews)
    for user in userImportantFeatures:
        userImportantFeatures[user]['name'] = user_dict[user]['name']
        userImportantFeatures[user]['stars'] = user_dict[user]['average_stars']
        
    if userInfo:
        out_file = open(outUserFile,"w")
        out_file.write(json.dumps(userImportantFeatures).encode('utf8', 'ignore'))
        out_file.close()
        
        out_file = open(outUserFile.replace('.json','.txt'),"w")
        for user in userImportantFeatures:
            out_file.write('%s\nstars = %s\n'%(userImportantFeatures[user]['name'].encode('utf8', 'ignore'),
                                                 str(userImportantFeatures[user]['stars']).encode('utf8', 'ignore')))
            out_file.write('%20sFeatures\tTFIDF\tFreq\tSentiment\n'%'')
            fA = featureAggregation(user_reviews[user])
            featuresInfo = list()
            for feature in userImportantFeatures[user]['tfidfDict']:
                if feature in userImportantFeatures[user]['featureFreq'] and feature in fA:
                    featuresInfo.append([userImportantFeatures[user]['tfidfDict'][feature],
                                         '%30s\t%s\t%s\t%s\n'%(feature,
                                                             str(userImportantFeatures[user]['tfidfDict'][feature]),
                                                             str(userImportantFeatures[user]['featureFreq'][feature]),
                                                             str(fA[feature]))])
            
            
            featuresInfo.sort(reverse=True)
            out_file.write(''.join([x[1] for x in featuresInfo]))           
            out_file.write('======================================================\n\n')                         
            
        out_file.close()
        
    logger.info('DONE')
    
    
    
    
    
if __name__ == '__main__':
    logger = logging.getLogger('signature')

    logging.basicConfig(format='%(asctime)s : %(name)-12s: %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))
    
    #path  = '../../data/'
    #path = '../../data/bing/American_New'
    path = '../../data/restaurants'
    importantFeatureIdentification(path+'/yelp_reviews_features.json',
                                   path+'/businessFeaturesAggregation.json',
                                   path+'/userFeaturesAggregation.json')
    