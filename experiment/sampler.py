import logging
import random

'''
Sample TRAIN and TEST
Input: reviews, prob
Output: list of reviews in TEST
'''


def sampler(reviews_filename, prob = 0.8):
    logger = logging.getLogger('signature.sampler')
    logger.info('starting sampling')
    
    #load reviews
    review_file = open(reviews_filename,"r")
    
    reviews = list()
    for counter, line in enumerate(review_file):
        reviews.append(line.strip())
        if not counter %10000:
            logger.info('%d reviews processed'%counter)
    review_file.close()
    
    random.shuffle(reviews)
    
    
    thres = len(reviews)*prob
    
    train_filename = reviews_filename.replace('.json','_train.json')
    test_filename = reviews_filename.replace('.json','_test.json')
    
    train_file = open(train_filename,"w")
    test_file = open(test_filename,"w")
    
    for counter, review in enumerate(reviews):
        if counter < thres:
            train_file.write(review+'\n')
        else:
            test_file.write(review+'\n')
    
    train_file.close()
    test_file.close()
    logger.info('DONE')



if __name__ == '__main__':
    #path  = '../../data/'
    path = '../../data/bing/American_New'
    