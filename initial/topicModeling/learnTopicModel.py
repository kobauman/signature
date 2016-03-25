from gensim import corpora, models, similarities
import json
import sys,os
sys.path.append('../')
import logging


import numpy as np


from utils.featuresStructure import featureStructureWorker

def learnTopicModel(infileName, dictFile, modelFile, descriptionFile, topic_num = 10):
    logger = logging.getLogger('signature.learnTopicModel')
    logger.info('starting learnTopicModel from %s'%infileName)
    fsw = featureStructureWorker()
    
    texts = list()
    #build corpus
    review_file = open(infileName,"r")
    for counter, line in enumerate(review_file):
        if not counter%10000:
            logger.debug('%d reviews loaded'%counter)
            
        # load review information
        review = json.loads(line.strip())
        reviewFeatures = fsw.getReviewFeaturesExistence(review['features'])
        
        text_plus = list()
        text_minus = list()
        
        for aspect in reviewFeatures:
            sent = np.average(reviewFeatures[aspect])
            if sent > 0:
                text_plus.append(aspect)
            elif sent < 0:
                text_minus.append(aspect)
        
        if len(text_plus):
            texts.append(text_plus)
        if len(text_minus):
            texts.append(text_minus)
        
    
    
    #build Dictionary
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=100, no_above=0.8)
    logger.info(dictionary)

    corpus_int = [dictionary.doc2bow(text) for text in texts]
    
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%m-%d,%H:%M:%S', level=logging.INFO)
    lda_model = models.ldamodel.LdaModel(corpus=corpus_int, id2word=dictionary, num_topics=topic_num, update_every=1, chunksize=10000, passes=30)
    lda_model.print_topics(20)
    
    output = open(descriptionFile,"w")
    for i, topic in enumerate(lda_model.show_topics(num_topics=100, num_words=15, log=False, formatted=True)):
        #print str(i)+"\t"+topic.encode("utf8")
        try:
            output.write(str(i)+"\t"+topic.decode('utf8', 'ignore')+"\n\n")
        except:
            try:
                output.write(str(i)+"\t"+topic[:30].decode('utf8', 'ignore')+"\n\n")
            except:
                output.write(str(i)+"\t"+"\n\n")
    output.close()
    
    dictionary.save(dictFile)
    lda_model.save(modelFile)
    
def learnTopics(path, topic_num = 10):
    model_path = path+'modelLDA/'
    try:
        os.stat(model_path)
    except:
        os.mkdir(model_path)
    infileName = path+'yelp_reviews_features_stat.json'
    dictFile = model_path+'dictionary_%d.lda'%topic_num
    modelFile = model_path+'model_%d.lda'%topic_num
    descriptionFile = model_path+'description_%d.txt'%topic_num
    learnTopicModel(infileName, dictFile, modelFile, descriptionFile, topic_num = 10)
    
    
if __name__ == '__main__':
    logger = logging.getLogger('signature')

    logging.basicConfig(format='%(asctime)s : %(name)-12s: %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))
    
    #path  = '../../data/'
    #path = '../../data/bing/American_New'
    path = '../../data/restaurants_topics_15/'
    learnTopics(path, topic_num = 15)