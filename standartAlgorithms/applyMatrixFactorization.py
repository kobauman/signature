import json
import logging


import numpy as np
import graphlab

def applyMatrixFactorization(trainReviews, model):
    logger = logging.getLogger('signature.aMF.applyMF')
    logger.info('starting applyMatrixFactorization for %d reviews'%len(trainReviews))
    
    data = {'id':[],'user':[],'item':[],'rating':[]}
    reviewDict = dict()
    for review in trainReviews:
        busID = review['business_id']
        userID = review['user_id']
        rating = review['stars']
            
        data['user'].append(userID)
        data['item'].append(busID)
        data['rating'].append(rating)
        ID = busID+'###'+userID
        data['id'].append(ID)
        reviewDict[ID] = review
        
    testData = graphlab.SFrame(data)
    testData['prediction'] = model.predict(testData)
    
    
    for i,ID in enumerate(testData['id']):
        reviewDict[ID]['MF_prediction'] = testData['prediction'][i]
    
    rmse = graphlab.evaluation.rmse(testData['rating'], testData['prediction'])
    logger.info('Score on test: %s'%str(rmse))
    
    return [reviewDict[i] for i in reviewDict]


def applyMF(path, model_num, limit = np.Inf):
    logger = logging.getLogger('signature.aMF')
    logger.info('starting applyMF')
    #get data
    r_file = path+'/yelp_reviews_features_test_pF_sent_agg.json'
    
    testReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        testReviews.append(json.loads(line.strip()))
    
    
    #load model
    model_path = path+'/regularModels/matrixFactorization_%d.model'%model_num
    model = graphlab.load_model(model_path)
    
    #run function
    reviewsPrediction = applyMatrixFactorization(testReviews, model)
    
    #save result
    outfile = open(path+'/yelp_reviews_features_test_pF_sent_agg_MF.json','w')
    for review in reviewsPrediction:
        outfile.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
    outfile.close()
    