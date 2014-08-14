import json
import logging
import pickle

import numpy as np
import graphlab as gl
import matplotlib.pyplot as plt


def learnMatrixFactorization(trainReviews):
    logger = logging.getLogger('signature.MF.learnMF')
    logger.info('starting learnMatrixFactorization from %d reviews'%len(trainReviews))
    
    data = {'user':[],'item':[],'rating':[]}
    for review in trainReviews:
        busID = review['business_id']
        userID = review['user_id']
        rating = review['stars']
            
        data['user'].append(userID)
        data['item'].append(busID)
        data['rating'].append(rating)
        
    learnData = gl.SFrame(data)
    
    model = gl.recommender.create(learnData,user_column='user',item_column='item',target_column='rating',
                                  method='matrix_factorization',n_factors=7,regularization=100,
                                  max_iterations=50,verbose=False)
        
    logger.info('Score on train: %s'%str(model.summary()['training_rmse']))
    
    return model

def learnMF(path):
    logger = logging.getLogger('signature.MF')
    logger.info('starting learnMF')
    #get data
    r_file = path+'/yelp_reviews_features_train.json'
    
    trainReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
#        if counter > 2000:
#            break
        trainReviews.append(json.loads(line.strip()))
    
    #run function
    model = learnMatrixFactorization(trainReviews)
    
    #save model
    model_path = path+'/regularModels/'
    model.save(model_path+'matrixFactorization_%d.model'%counter)