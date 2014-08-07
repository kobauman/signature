import json
import re, glob
import logging

'''
5) Combine all data together
Input: sentenceID part_start part_end feature sentiment ([-1,0,1])
Output: review -> {setence_num:{featureID:sentiment,..}, }
'''

train_data_path = "../../../yelp/data/splitting_reviews/"

def Filenames(directory):
    if not directory.endswith("/"):
            directory+="/"
    return glob.glob(directory+"*")

def getFeatures(path):
    logger = logging.getLogger('signature.gF')
    logger.info('starting getFeatures')
    files = [x for x in Filenames(path+'output/') if 'output_' in x and x.endswith('.txt')]
    logger.info('Files: '+';\n'.join(files))
    features = dict()
    for filename in files:
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
            if not counter%1000:
                logger.debug('%d sentencies'%counter)
            reviewID = sentenceID/10000
            sentenceNum = sentenceID%10000
            if reviewID not in reviews_set:
                features[reviewID] = dict()
                reviews_set.add(reviewID)
            features[reviewID][sentenceNum] = features[reviewID].get(sentenceNum, {})
            features[reviewID][sentenceNum][featureID] = sentiment
        feature_file.close()
    return features  

def preProcessReviews(review_features, outfilename, limit = 100):
    logger = logging.getLogger('signature.pPR')
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
            del review['text']
            out_file.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
        if not counter %10000:
            logger.info('%d reviews processed'%counter)
            
    review_file.close()
    out_file.close()





if __name__ == '__main__':
    #path  = '../../data/'
    path = '../../data/bing/American_New'
    review_features = getFeatures(path)
    
    preProcessReviews(review_features, path+'/yelp_reviews_features.json', limit = 1000000)