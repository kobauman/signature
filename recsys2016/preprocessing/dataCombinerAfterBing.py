import os,sys

import pandas as pd
import numpy as np

import json
import re, glob
import logging

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


def getAspects(datafiles):
    logger = logging.getLogger('signature.getFeatures')
    logger.info('starting getFeatures')
    files = [x for x in datafiles if x.endswith('.txt')]
    logger.info('Files: '+';\n'.join(files))
    aspects = dict()
    for filename in files:
        logger.debug('Loading: %s'%filename)
        with open(filename,'r') as aspect_file: 
            for counter, line in enumerate(aspect_file):
                l = line.strip().split('|')
                try:
                    if l[1].strip() == 'F':
                        sentenceID = int(l[0].replace(')',''))
                        aspectID = l[4].strip()
                        sentiment = l[5]
                    else:
                        continue
                except:
                    print(line)
    #            if not counter%1000:
    #                logger.debug('%d sentencies'%counter)
                reviewID = sentenceID//10000
                sentenceNum = sentenceID%10000
    #            print(sentenceID,reviewID, sentenceNum)
                aspects[reviewID] = aspects.get(reviewID,{})
                aspects[reviewID][sentenceNum] = aspects[reviewID].get(sentenceNum, {})
                aspects[reviewID][sentenceNum][aspectID] = sentiment
    return aspects  






def preProcessReviews(original_dataset, r_aspects, output, limit = 100):
    #preProcessReviews_examples(review_features, outfilename, limit = 100):
    logger = logging.getLogger('signature.preProcessReviews_examples')
    logger.info('starting preProcessReviews_examples')
    
    aspect_parents = dict()
    def asp_par(aspect):
        result = list()
        a = aspect.split('_')
        for i in range(1,len(a)+1):
            result.append('_'.join(a[:i]))
        return result
    
    #load reviews
    
    out_file = open(output,"w")
    reviews = []
    print(original_dataset)
    with open(original_dataset,"r") as review_file:
        for counter, line in enumerate(review_file):
            if counter > limit:
                break
            review = json.loads(line)
            #businessID = business_dict.get(review["business_id"],None)
            #userID = user_dict.get(review["user_id"],None)
            
            if counter in r_aspects:
                review['aspects'] = r_aspects[counter].copy()
                review['ID'] = counter
                #textFeatures, sentences = getReviewStat(review['text'])
                #print(textFeatures)
                
                reviews.append(review)
                
                aspects = dict()
                for sent in review['aspects']:
                    for asp in review['aspects'][sent]:
                        if asp not in aspect_parents:
                            aspect_parents[asp] = asp_par(asp)
                        for a in aspect_parents[asp]:
                            aspects[a] = aspects.get(a, [])
                            aspects[a].append(int(review['aspects'][sent][asp]))
                
                for asp in aspects:
                    review[asp] = 1
                    review[asp+'sent'] = int(np.sign(np.average(aspects[asp])))
                
#                 print(review['aspects'],aspects)
#                 print(review)
                
                # delete heavy fields
                del review['text']
                del review['aspects']
                
    #             review['sentences'] = sentences
    #             review['textFeatures'] = textFeatures
                
    #             for i, sent in enumerate(sentences):
    # #                print(review['features'])
    #                 if i in review['aspects']:
    #                     out_file.write('%s\n%s\n\n'%(sent,str(review['features'][i])))
            if not counter %1000:
                logger.info('%d reviews processed'%counter)
    #        if counter > 100:
    #            break
            
    result = pd.DataFrame(reviews)
    result.to_csv(out_file, index=False)





if __name__ == '__main__':
    logger = logging.getLogger('signature')
    logging.basicConfig(format='%(asctime)s : %(name)-12s: %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))
    
    
    
    path = '../../../data/'
    
#     appName = 'beautySpa'
#     appName = 'hotel'
    appName = 'restaurant'
    
    # 1) get list of features
    bing_path = os.path.join(path,'BING_DATASET', appName)
    bing_files = Filenames(bing_path)
    app_aspects = getAspects(bing_files)
    
    
    # 2) combine reviews with aspects
    original_dataset = os.path.join(path,'YELP_DATASET/yelp_dataset_challenge_academic_dataset',
                                    'yelp_academic_dataset_review.json')
    
    
    output = os.path.join(path,'BING_DATASET', appName+'_data.csv')
    
    preProcessReviews(original_dataset, app_aspects, output, limit = 10000000)