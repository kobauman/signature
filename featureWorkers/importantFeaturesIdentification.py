import json
import sys
sys.path.append('../')
import logging

from utils.featureItemImportance import featureImportance


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



def importantFeatureIdentification(infileName, outBusinesFile, outUserFile, busInfo = False, userInfo = False, limit = 100000000):
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
        if not counter%10000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        # load review information
        review = json.loads(line)
        businessID = review['business_id']
        userID = review['user_id']
        
        # fill up dictionaries of reviews
        #if businessID in business_dict:
        bus_reviews[businessID] = bus_reviews.get(businessID,[])
        bus_reviews[businessID].append(review)
        
        #if userID in user_dict:
        user_reviews[userID] = user_reviews.get(userID,[])
        user_reviews[userID].append(review)    
    review_file.close()
    
    logger.info('Important feature identification for %d BUSINESSES'%len(bus_reviews))
    busImportantFeatures = featureImportance(bus_reviews, False)
    for bus in busImportantFeatures:
        if bus in business_dict:
            busImportantFeatures[bus]['categories'] = business_dict[bus]['categories']
            busImportantFeatures[bus]['name'] = business_dict[bus]['name']
            busImportantFeatures[bus]['stars'] = business_dict[bus]['stars']
        else:
            busImportantFeatures[bus]['categories'] = None
            busImportantFeatures[bus]['name'] = None
            busImportantFeatures[bus]['stars'] = None
        
    
    out_file = open(outBusinesFile,"w")
    out_file.write(json.dumps(busImportantFeatures).encode('utf8', 'ignore'))
    out_file.close()
        
    if busInfo:   
        out_file = open(outBusinesFile.replace('.json','.txt'),"w")
        for b, bus in enumerate(busImportantFeatures):
            out_file.write('%s\t%s\nstars = %s\n'%(busImportantFeatures[bus]['name'].encode('utf8', 'ignore'),
                                                 str(busImportantFeatures[bus]['categories']).encode('utf8', 'ignore'),
                                                 str(busImportantFeatures[bus]['stars']).encode('utf8', 'ignore')))
            out_file.write('%20sFeatures\tTFIDF\tFreq\tSentiment\n'%'')
            featuresInfo = list()
            for feature in busImportantFeatures[bus]['tfidfDict']:
                if feature in busImportantFeatures[bus]['featureFreq']:
                    featuresInfo.append([busImportantFeatures[bus]['tfidfDict'][feature],
                                         '%30s\t%s\t%s\t%s\n'%(feature,
                                                             str(busImportantFeatures[bus]['tfidfDict'][feature]),
                                                             str(busImportantFeatures[bus]['featureFreq'][feature]),
                                                             str(busImportantFeatures[bus]['sentiment'][feature]))])
            
            
            featuresInfo.sort(reverse=True)
            out_file.write(''.join([x[1] for x in featuresInfo]))           
            out_file.write('======================================================\n\n')                         
            
            if not b%1000:
                logger.debug('%d business summaries completed'%b)
        out_file.close()
    
    
    
    logger.info('Important feature identification for %d USERS'%len(user_reviews))
    userImportantFeatures = featureImportance(user_reviews, False)
    for user in userImportantFeatures:
        if user in user_dict:
            userImportantFeatures[user]['name'] = user_dict[user]['name']
            userImportantFeatures[user]['stars'] = user_dict[user]['average_stars']
        else:
            userImportantFeatures[user]['name'] = None
            userImportantFeatures[user]['stars'] = None
        
    
    out_file = open(outUserFile,"w")
    out_file.write(json.dumps(userImportantFeatures).encode('utf8', 'ignore'))
    out_file.close()
        
    if userInfo:   
        out_file = open(outUserFile.replace('.json','.txt'),"w")
        for u, user in enumerate(userImportantFeatures):
            try:
                out_file.write('%s\nstars = %s\n'%(str(userImportantFeatures[user]['name']).encode('utf8', 'ignore'),
                                                 str(userImportantFeatures[user]['stars']).encode('utf8', 'ignore')))
            except:
                pass
            out_file.write('%20sFeatures\tTFIDF\tFreq\tSentiment\n'%'')
            featuresInfo = list()
            for feature in userImportantFeatures[user]['tfidfDict']:
                if feature in userImportantFeatures[user]['featureFreq']:
                    featuresInfo.append([userImportantFeatures[user]['tfidfDict'][feature],
                                         '%30s\t%s\t%s\t%s\n'%(feature,
                                                             str(userImportantFeatures[user]['tfidfDict'][feature]),
                                                             str(userImportantFeatures[user]['featureFreq'][feature]),
                                                             str(userImportantFeatures[user]['sentiment'][feature]))])
            
            
            featuresInfo.sort(reverse=True)
            out_file.write(''.join([x[1] for x in featuresInfo]))           
            out_file.write('======================================================\n\n')                         
            
            if not u%1000:
                logger.debug('%d user summaries completed'%u)
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
    