import json
import logging


import numpy as np

from utils.featuresStructure import featureStructureWorker





def matchBUprofiles(fsw, busProfile, userProfile):
    fsw = featureStructureWorker()
    
    matchScore = 0.0
    for feature in fsw.featureIdicator:
        if not fsw.featureIdicator[feature]:
            continue
        
        if feature not in userProfile['featureFreq'] or feature not in busProfile['featureFreq']:
            continue
        
        if userProfile['featureFreq'][feature] > 10 and userProfile['sentiment'][feature][1] > 1:
            pass
        else:
            continue
        
        if busProfile['featureFreq'][feature] > 10 and busProfile['sentiment'][feature][1] > 5:
            pass
        else:
            continue
        
        sentiment = busProfile['sentiment'][feature][0]
        userImp = userProfile['tfidfDict'].get(feature,0.0)
        busImp = busProfile['tfidfDict'].get(feature,0.0)
        matchScore += userImp*busImp*sentiment
        
    return matchScore
        



def matchProfiles(path, limit = np.Inf):
    logger = logging.getLogger('signature.matchProfiles')
    logger.info('starting matchProfiles')
    #get data
    b_file = path+'/businessFeaturesAggregation_train.json'
    u_file = path+'/userFeaturesAggregation_train.json'
    r_file = path+'/yelp_reviews_features_test_pF_sent_agg_MF.json'
    
    busImportantFeatures = json.loads(open(b_file,'r').readline())
    logger.info('Important BUSINESS Features loaded')
    userImportantFeatures = json.loads(open(u_file,'r').readline())
    logger.info('Important USER Features loaded')
    fsw = featureStructureWorker()
    
    outfile = open(path+'/yelp_reviews_features_test_pF_sent_agg_MF_match.json','w')
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        review = json.loads(line.strip())
        userID = review['user_id']
        busID = review['business_id']
        if busID in busImportantFeatures and userID in userImportantFeatures:
            review['match'] = matchBUprofiles(fsw, busImportantFeatures[busID], userImportantFeatures[userID])
        else:
            review['match'] = -1000000
        
        outfile.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
    outfile.close()