import json
import logging


import numpy as np
import graphlab as gl


def applyMatrixFactorization(trainReviews, model):
    logger = logging.getLogger('signature.aMF.applyMF')
    logger.info('starting applyMatrixFactorization for %d reviews'%len(trainReviews))
    
    data = {'user':[],'item':[],'rating':[]}
    for review in trainReviews:
        busID = review['business_id']
        userID = review['user_id']
        rating = review['stars']
            
        data['user'].append(userID)
        data['item'].append(busID)
        data['rating'].append(rating)
        
    testData = gl.SFrame(data)
        
    prediction = model.score(testData)
    
    rmse = gl.evaluation.rmse(testData['rating'], prediction)
    
    logger.info('Score on test: %s'%str(rmse))
    
    data['prediction'] = list(prediction)
    return data


def applyMF(path, modelPath):
    logger = logging.getLogger('signature.aMF')
    logger.info('starting applyMF')
    #get data
    r_file = path+'/yelp_reviews_features_test.json'
    
    trainReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
#        if counter > 2000:
#            break
        trainReviews.append(json.loads(line.strip()))
    
    
    #load model
    model = gl.load_model(modelPath)
    
    #run function
    pred = applyMatrixFactorization(trainReviews, model)
    
    #save prediction
    output = open(path + '/predictions/matrixFactorization.csv', 'w')
    lines = list()
    for i in range(len(pred['user'])):
        #print i
        line = '%s,%s,%d,%.3f'%(pred['user'][i],pred['item'][i],pred['rating'][i],pred['prediction'][i])
        lines.append(line)
        
    output.write('\n'.join(lines))
    output.close()
    
    