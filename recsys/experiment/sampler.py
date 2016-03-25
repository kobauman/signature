import logging
import random
import json
import os

'''
Sample TRAIN and TEST
Input: reviews, prob
Output: list of reviews in TEST
'''


def sampler(path, reviews_filename, probs = [0.4, 0.8], busThres = 0, userThres = 0):
    logger = logging.getLogger('signature.sampler')
    logger.info('starting sampling')
    
    #load reviews
    review_file = open(reviews_filename,"r")
    
    bus_info = dict()
    user_info = dict()
    
    reviews = list()
    for counter, line in enumerate(review_file):
        reviews.append(json.loads(line))
        busId = reviews[-1]['business_id']
        userId = reviews[-1]['user_id']
        bus_info[busId] = bus_info.get(busId,0)
        bus_info[busId]+=1
        
        user_info[userId] = user_info.get(userId,0)
        user_info[userId]+=1
        
        if not counter %10000:
            logger.info('%d reviews processed'%counter)
            
    review_file.close()
    
    r_num = len(reviews)
    #clean by business
    good_bus = set([bus for bus in bus_info if bus_info[bus] > busThres])
    reviews = [review for review in reviews if review['business_id'] in good_bus]
    
    good_user = set([user for user in user_info if user_info[user] > userThres])
    reviews = [review for review in reviews if review['user_id'] in good_user]
    
    logger.info('Num of businesses before = %d, after = %d'%(len(bus_info),len(good_bus)))
    logger.info('Num of users before = %d, after = %d'%(len(user_info),len(good_user)))
    logger.info('Num of reviews before = %d, after = %d'%(r_num,len(reviews)))
    
    
    
    #shuffle
    random.shuffle(reviews)
    
    
    
    thres1 = len(reviews)*probs[0]
    thres2 = len(reviews)*probs[1]
    
    train_filename = reviews_filename.replace('_features.json','_train.json')
    stat_filename = reviews_filename.replace('_features.json','_stat.json')
    extrain_filename = reviews_filename.replace('_features.json','_extrain.json')
    test_filename = reviews_filename.replace('_features.json','_test.json')
    
    train_file = open(train_filename,"w")
    stat_file = open(stat_filename,"w")
    extrain_file = open(extrain_filename,"w")
    test_file = open(test_filename,"w")
    
    
    counters = [0,0,0,0]
    for counter, review in enumerate(reviews):
        review = json.dumps(review)
        if counter < thres2:
            train_file.write(review+'\n')
            counters[0] += 1
            
        if counter < thres1:
            stat_file.write(review+'\n')
            counters[1] += 1
            
        elif counter < thres2:
            extrain_file.write(review+'\n')
            counters[2] += 1
            
        else:
            test_file.write(review+'\n')
            counters[3] += 1
            
    train_file.close()
    stat_file.close()
    extrain_file.close()
    test_file.close()
    logger.info('DONE %s'%str(counters))
    
    try:
        os.stat(path+'results/')
    except:
        os.mkdir(path+'results/')
    
    outfile = open(path+'results/Numbers_stat.txt','w')
    outfile.write('Businesses only with > %d reviews\nUsers only with > %d reviews'%(busThres,userThres))
    outfile.write('\nTrain: %d,\n Stat: %d,\nExtrain: %d,\nTest: %d'%(counters[0],counters[1],counters[2],counters[3]))
    outfile.close()

    