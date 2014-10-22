import json
import logging
import os

import numpy as np
import graphlab


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
        
    learnData = graphlab.SFrame(data)
    
    
#    MFmodel = graphlab.recommender.factorization_recommender.create(learnData,user_id='user',item_id='item',
#                                                                    target='rating',num_factors=10,
#                                                                    #regularization=100,#binary_targets=True,
#                                                                    max_iterations=50,verbose=False)
    
    MFmodel = graphlab.recommender.create(learnData,user_id='user',item_id='item',
                                           target='rating',#n_factors=10,
                                           #regularization=100,#binary_targets=True,
                                           #max_iterations=50,verbose=False,
                                           method = 'matrix_factorization')
     
       
    #logger.info('Score on train: %s'%str(model.summary()['training_rmse']))
    
    return MFmodel

def learnMF(path, limit = np.Inf):
    logger = logging.getLogger('signature.MF')
    logger.info('starting learnMF')
    #get data
    r_file = path+'/yelp_reviews_features_train.json'
    
    trainReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        trainReviews.append(json.loads(line.strip()))
    
    #run function
    model = learnMatrixFactorization(trainReviews)
    
    #save model
    model_path = path+'/regularModels/'
    try:
        os.stat(model_path)
    except:
        os.mkdir(model_path)
    model.save(model_path+'matrixFactorization_%d.model'%counter)
    
    return counter