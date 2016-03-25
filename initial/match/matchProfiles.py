import json
import logging


import numpy as np

from utils.featuresStructure import featureStructureWorker





def matchBUprofiles(fsw, featureWeights, busProfile, userProfile):
    fsw = featureStructureWorker()
    
    matchScore = 0.0
    num = 0
    for feature in fsw.featureIdicator:
        if not fsw.featureIdicator[feature]:
            continue
        
        if feature not in userProfile['featureFreq'] and feature not in busProfile['featureFreq']:
            continue
        
        if userProfile['featureFreq'].get(feature,0) > 10 and userProfile['sentiment'].get(feature,[0,0])[1] > 1:
            pass
        else:
            continue
        
        if busProfile['featureFreq'].get(feature,0) > 10 and busProfile['sentiment'].get(feature,[0,0])[1] > 5:
            pass
        else:
            continue
            
        
        
        
        sentiment = busProfile['sentiment'][feature][0]
#        userImp = userProfile['tfidfDict'].get(feature,0.0)
#        busImp = busProfile['tfidfDict'].get(feature,0.0)
        userImp = userProfile['featureFreq'].get(feature,0.0)
        busImp = busProfile['featureFreq'].get(feature,0.0)
        weight = featureWeights[feature]
        coeff = 1.0
        
        if sentiment < weight:
            coeff = 2.0
        
#        if busImp > 80:
#            userImp = max(userImp,busImp)
        
        
        matchScore += userImp*busImp*sentiment*weight*coeff
        num += 1.0
    return num, matchScore
        



def matchProfiles(path, limit = np.Inf):
    logger = logging.getLogger('signature.matchProfiles')
    logger.info('starting matchProfiles')
    #get data
    b_file = path+'businessFeaturesAggregation_train.json'
    u_file = path+'userFeaturesAggregation_train.json'
    r_file = path+'yelp_reviews_test_predictions.json'
    
    busImportantFeatures = json.loads(open(b_file,'r').readline())
    logger.info('Important BUSINESS Features loaded')
    userImportantFeatures = json.loads(open(u_file,'r').readline())
    logger.info('Important USER Features loaded')
    fsw = featureStructureWorker()
    
    #load featureWeights
    infile = open(path+'featureWeights.json','r')
    featureWeights = json.loads(infile.readline().strip())
    infile.close()
    
    nums = list()
    reviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        review = json.loads(line.strip())
        userID = review['user_id']
        busID = review['business_id']
        if busID in busImportantFeatures and userID in userImportantFeatures:
            num, score = matchBUprofiles(fsw, featureWeights,
                                              busImportantFeatures[busID], 
                                              userImportantFeatures[userID])
            nums.append(num)
        else:
            score = -1000000
        
        review['rating_prediction'] = review.get('rating_prediction', {})
        review['rating_prediction']['match'] = score
        
        reviews.append(review)
        
    outfile = open(path+'yelp_reviews_test_predictions.json','w')    
    for review in reviews:
        outfile.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
    outfile.close()
    #print nums
    #print len(nums)
    print 'AVERAGE NUMBER OF FEATURES = %f'%np.average(num)