import json
import re, glob
import logging
import math
from textblob import TextBlob
from io import StringIO

'''
0) Combine all data together
Input: sentenceID part_start part_end feature sentiment ([-1,0,1])
Output: review -> {setence_num:{featureID:sentiment,..}, }
'''

train_data_path = "../../../yelp/data/splitting_reviews/"

def Filenames(directory):
    if not directory.endswith("/"):
            directory+="/"
    return glob.glob(directory+"*")


def divide(a,b):
    try:
        #suppose that number2 is a float
        return round(float(a)/float(b),3)
    except ZeroDivisionError:
        return -10.0

def getReviewStat(text):
    features_list = list()
    textB = TextBlob(text)
    #'logLen(%d)'
    features_list.append(math.log(len(textB.sentences)+1))
    #'logLenWords(%d)'
    features_list.append(math.log(len(textB.words)+1))
    
    VBDsum = sum([1 for tag in textB.tags if tag[1] == 'VBD'])
    Vsum = sum([1 for tag in textB.tags if tag[1].startswith('VB')])
    #'logVBDsum(%d)'
    features_list.append(math.log((VBDsum+1),2))
    #'logVsum(%d)'
    features_list.append(math.log((Vsum+1),2))
    ratio = divide(VBDsum*10,Vsum)
    #'intVBD/Vsum(%d)'
    features_list.append(ratio)
    
    sentences = [str(x) for x in textB.sentences]
    
    return features_list, sentences



def getFeatures(path):
    logger = logging.getLogger('signature.getFeatures')
    logger.info('starting getFeatures')
    files = [x for x in Filenames(path+'bing_output/') if 'output_' in x and x.endswith('.txt')]
    logger.info('Files: '+';\n'.join(files))
    features = dict()
    for filename in files:
        logger.debug('Loading: %s'%filename)
        reviews_set = set()
        feature_file = open(filename,'r')
        for counter, line in enumerate(feature_file):
            l = line.strip().split('|')
            if l[1].strip() == 'F':
                sentenceID = int(l[0].replace(')',''))
                featureID = l[4].strip()
                sentiment = l[5]
            else:
                continue
#            if not counter%1000:
#                logger.debug('%d sentencies'%counter)
            reviewID = sentenceID//10000
            sentenceNum = sentenceID%10000
#            print(sentenceID,reviewID, sentenceNum)
            if reviewID not in reviews_set:
                features[reviewID] = dict()
                reviews_set.add(reviewID)
            features[reviewID][sentenceNum] = features[reviewID].get(sentenceNum, {})
            features[reviewID][sentenceNum][featureID] = sentiment
        feature_file.close()
    #print features.keys()
    return features  

def preProcessReviews(review_features, outfilename, limit = 100):
    logger = logging.getLogger('signature.preProcessReviews')
    logger.info('starting preProcessReviews')
    
    #load reviews
    review_file = open(train_data_path + "yelp_training_set_review.json","r")
       
    out_file = open(outfilename,"w")
    for counter, line in enumerate(review_file):
        if counter > limit:
            break
        review = json.loads(line)
        #businessID = business_dict.get(review["business_id"],None)
        #userID = user_dict.get(review["user_id"],None)
        
        if counter in review_features:
            review['features'] = review_features[counter].copy()
            review['ID'] = counter
            textFeatures, sentences = getReviewStat(review['text'])
            #print(textFeatures)
            #del review['text']
            
            review['sentences'] = sentences
            review['textFeatures'] = textFeatures
            
            out_file.write(json.dumps(review)+'\n')
        if not counter %1000:
            logger.info('%d reviews processed'%counter)
#        if counter > 100:
#            break
            
    review_file.close()
    out_file.close()




def preProcessReviews_examples(review_features, outfilename, limit = 100):
    logger = logging.getLogger('signature.preProcessReviews_examples')
    logger.info('starting preProcessReviews_examples')
    
    #load reviews
    review_file = open(train_data_path + "yelp_training_set_review.json","r")
       
    out_file = open(outfilename,"w")
    for counter, line in enumerate(review_file):
        if counter > limit:
            break
        review = json.loads(line)
        #businessID = business_dict.get(review["business_id"],None)
        #userID = user_dict.get(review["user_id"],None)
        
        if counter in review_features:
            review['features'] = review_features[counter].copy()
            review['ID'] = counter
            textFeatures, sentences = getReviewStat(review['text'])
            #print(textFeatures)
            #del review['text']
            
            review['sentences'] = sentences
            review['textFeatures'] = textFeatures
            
            for i, sent in enumerate(sentences):
#                print(review['features'])
                if i in review['features']:
                    out_file.write('%s\n%s\n\n'%(sent,str(review['features'][i])))
        if not counter %1000:
            logger.info('%d reviews processed'%counter)
#        if counter > 100:
#            break
            
    review_file.close()
    out_file.close()





if __name__ == '__main__':
    #path  = '../../data/'
    path = '../../data/bing/American_New'
    review_features = getFeatures(path)
    
    preProcessReviews(review_features, path+'/yelp_reviews_features.json', limit = 1000000)