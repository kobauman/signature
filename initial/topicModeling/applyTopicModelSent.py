from gensim import corpora, models, similarities
import json
import logging
import sys
sys.path.append('../')

import numpy as np
from utils.featuresStructure import featureStructureWorker

def topTopics(topics_list, sign = 1):
    rev = [[x[1],x[0]] for x in topics_list]
    rev.sort(reverse=True)
    thres = max(rev[0][0]/3.0, 0.11)
    thres = 0
    result = dict()
    for topic in rev:
        if topic[0] > thres:
            result[topic[1]] = sign*round(topic[0],4)
    return result.copy()
    
    
def applyTopicModel(logger, path, topic_num):
    stat_file = path+'yelp_reviews_features_stat.json'
    train_file = path+'yelp_reviews_features_train.json'
    extrain_file = path+'yelp_reviews_features_extrain.json'
    test_file = path+'yelp_reviews_features_test.json'
    
    #load model
    model_path = path+'modelLDA/'
    dictionary = corpora.Dictionary.load(model_path+'dictionary_%d.lda'%topic_num)
    logger.info("Dictionary loaded from: "+ model_path+'dictionary_%d.lda'%topic_num)

    lda_model = models.ldamodel.LdaModel.load(model_path+'model_%d.lda'%topic_num)
    logger.info("Model loaded from:" + model_path+'model_%d.lda'%topic_num)
    
    
    files = [stat_file,train_file,extrain_file,test_file]
    fsw = featureStructureWorker()
    
    for infile in files:
        reviews = list()
        for counter, line in enumerate(open(infile,'r')):
            if not counter%10000:
                logger.debug('%d reviews loaded'%counter)
            #print infile, line
            # load review information
            review = json.loads(line.strip())
            reviews.append(review)
            
#        outfile = open(infile.replace('.json','_old.json'),'w')
#        for review in reviews:
#            outfile.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
#            outfile.close()
        
        
#        outfile = open(infile,'w')
#        outname = infile.replace('.json','_old.json')
        outname = infile
        print outname
        outfile = open(outname,'w')
        for counter, review in enumerate(reviews):
            if not counter%1000:
                logger.debug('%d reviews loaded'%counter)
            
            
            text = list()
            for sentence in review['features']:
                for aspect in review['features'][sentence]:
                    text.append(aspect+'_%s'%review['features'][sentence][aspect].strip())
        
            
            topics = lda_model[dictionary.doc2bow(text)]
            
            res = dict()
            if len(topics):
                res['1'] = topTopics(topics)
            #print topics, res
            
            if 'features_sent' not in review:
                review['features_sent'] = review['features'].copy()
            review['features'] = res.copy()
            
            
            outfile.write(json.dumps(review).encode('utf8', 'ignore')+'\n')
            
#            if counter > 10:
#                break
        outfile.close()
        
            
if __name__ == '__main__':
    logger = logging.getLogger('signature')

    logging.basicConfig(format='%(asctime)s : %(name)-12s: %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))
    
    #path  = '../../data/'
    #path = '../../data/bing/American_New'
    #path = '../../data/restaurants_topics_15/'
    path = '../../data/restaurants_topics_sent/'
    topic_num = 20
    applyTopicModel(logger, path, topic_num)