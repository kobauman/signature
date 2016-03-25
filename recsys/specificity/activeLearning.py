from textblob.classifiers import NaiveBayesClassifier
import random
import json
import re
import pickle
import pandas as pd
import logging


def feature_extractor(element):
    features = dict()
    features['logLen'] = int(element[1][0])
    features['logLenWords'] = int(element[1][1])
    features['logVBDsum'] = int(element[1][2])
    features['logVsum'] = int(element[1][3])
    features['intVBD/Vsum'] = int(element[1][4])
    
    return features


def sp_prob(model, review):
    return round(model.prob_classify([re.sub(r'[^\x00-\x7f]', r' ', review['text']), review['textFeatures']]).prob('s'), 2)




def activeLearning(NAME, datapath, infile, iterations = 3, portion = 10):
    logger = logging.getLogger('signature.activeLearning')
    logger.info('Active learning model building')
    
    #load data
    review_file = open(infile,"r")
    
    #convert to appropriate format
    review_corpus = list()
    for i, line in enumerate(review_file):
        try:
            #filter out non-ascii simbols
            review = json.loads(line)
            review_corpus.append([re.sub(r'[^\x00-\x7f]', r' ', review['text']), review['textFeatures']])
        except:
            logger.error(review['text'])
            continue
    review_file.close()
    logger.info('Data converted - %d reviews'%len(review_corpus))
    
    
    #Shuffle dataset
    #random.seed(1)
    random.shuffle(review_corpus)
    
    try:
        current_train = json.loads(open(datapath + '%s_current_train.json'%NAME,'r').read())
    except:
        current_train = list()
    
    for t in current_train:
        try:
            review_corpus.remove(t[0])
        except:
            pass
    
    logger.info("Len(current_train) = %d"%len(current_train))
    
    '''
    Prepare first portion
    '''
    if len(current_train) > 10:
        #train model
        cl = NaiveBayesClassifier(current_train, feature_extractor=feature_extractor)
        
        #prepare next portion
        ratio = float(sum([int(x[1] == 'g') for x in current_train]))/len(current_train)
        #ratio = 0.5
        logger.info('ratio = %.3f\nclassifying train set ...'%ratio)
        train_classify = [[0.1*random.random() + abs(int(cl.classify(t)=='s')-ratio),t] for t in review_corpus[:1000]]
        train_classify.sort()
        reviews_portion = train_classify[:portion]
    
    else:
        reviews_portion = [y for y in enumerate(review_corpus[:portion])]

    
    '''
    main iterations of active learning
    '''
    for iteration in range(iterations):
        #ask for labels
        for p in range(len(reviews_portion)):
            var = input('''\n\n%s \n(%f)\nPlease give the label to the review 
(g - generic / s - specific): '''%(reviews_portion[p][1][0],reviews_portion[p][0]))
            
            if var.lower().startswith('g'):
                label = 'g'
            elif var.lower().startswith('s'):
                label = 's'
            elif var.lower().startswith('x'):
                logger.info('Finish')
                break
            else:
                logger.info('Bad label')
                continue
        
            #prepare train set
            current_train.append((reviews_portion[p][1],label))
            review_corpus.remove(reviews_portion[p][1])
        
        #train model
        cl = NaiveBayesClassifier(current_train, feature_extractor=feature_extractor)
        
        #prepare next portion
        ratio = float(sum([int(x[1] == 'g') for x in current_train]))/len(current_train)
        #ratio = 0.5
        logger.info('ratio = %.3f\nclassifying train set ...'%ratio)
        train_classify = [[0.1*random.random() + abs(int(cl.classify(t)=='s')-ratio),t] for t in review_corpus[:1000]]
        train_classify.sort()
        reviews_portion = train_classify[:portion]
        
        logger.info('Iteration: %d (%d items), Accuracy on train = %.2f'%(iteration,len(current_train),100*cl.accuracy(current_train)))
        
        current_train_out = open(datapath+'%s_current_train.json'%NAME,'w')
        current_train_out.write(json.dumps(current_train))
        current_train_out.close()
        
    
    cl.show_informative_features(10)
    
    
    
    
    #test
    random.shuffle(current_train)
    thres = int(0.8*len(current_train))
    train_self = current_train[:thres]
    test_self = current_train[thres:]
    cl_test =  NaiveBayesClassifier(train_self, feature_extractor=feature_extractor)
    acc_str = 'Accuracy on test = %.2f with %d items in testset and %d items in trainset'%(100*cl_test.accuracy(test_self),
                                                                                           len(test_self),len(train_self))
    logger.info(acc_str)
    message = list()
    message.append(acc_str)
        
    #saving model
    pickle.dump(cl, open(datapath+ '%s_active_learning.model'%NAME, "wb" ) )
    
    
    return '\n'.join(message)



def apply_model(NAME, datapath, infile, outfile):
    logger = logging.getLogger('signature.specificityApply')
    logger.info('Apply model')
    
    #load model
    cl = pickle.load( open(datapath + '%s_active_learning.model'%NAME, "rb" ) )
    logger.info('Model loaded')
    
    #apply model
    review_file = open(infile,"r")
    review_out = list()
    
    
    for i, line in enumerate(review_file):
        try:
            #filter out non-ascii simbols
            review = json.loads(line)
        except:
            logger.error(review['text'])
            continue
        
        prediction = sp_prob(cl, review)
        if prediction > 0.5:
            del review['text']
            review_out.append(json.dumps(review))
          
    review_file.close()
    
    review_outfile = open(outfile,"w")
    review_outfile.write('\n'.join(review_out))
    review_outfile.close()
    
    message = '%d specific reviews identified from %d reviews'%(len(review_out), i)
    logger.info(message)
    
    return message
    

