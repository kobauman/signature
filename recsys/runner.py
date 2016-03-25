'''
This script runs all steps of the signature construction process
'''

#the name of the dataset
NAME = 'restaurants'
#NAME = 'restaurants_all'
#NAME = 'restaurants_a'

#NAME = 'beautyspa'
datapath = '../../data_recsys/'

import sys
sys.path.append('../')

import logging
import time
import datetime
import os


from preparation.dataCombiner import *
from specificity.activeLearning import activeLearning, apply_model
from experiment.sampler import sampler
from featureWorkers.importantFeaturesIdentification import importantFeatureIdentification
from featureWorkers.learnFeaturesExistence import learnFE
from featureWorkers.applyFeaturesExistence import applyFE
from featureWorkers.computeStat import computeStat
from featureWorkers.aspectStat import aspectStat
from featureWorkers.pairCompare import pairCompare
from featureWorkers.predictAll import predictAll

from dependence.dependence import aspectDependence,userItemStats

if __name__ == '__main__':
    path = datapath + NAME + '/'
    
    logger = logging.getLogger('signature')
    logfile = datapath+'/log/%d_%s.log'%(int(time.time()),NAME)
    logging.basicConfig(filename = logfile, format='%(asctime)s : %(name)-12s: %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))
    
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s : %(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    '''Write statistics and results'''
    try:
        os.stat(path+'results/')
    except:
        os.mkdir(path+'results/')
    
    
    
    '''
    0) combine data from Bing with POS tagged data
    dataCombiner.py
    '''
#    outfile = path+'reviews_features.json'
#    review_features = getFeatures(path)
#    preProcessReviews(review_features, outfile, limit = 300000)
    
#    '''
#    0*) combine data from Bing with POS tagged data and build examples
#    dataCombiner_example.py
#    '''
#    outfile = path+'reviews_features.txt'
#    review_features = getFeatures(path)
#    preProcessReviews_examples(review_features, outfile, limit = 30000)
#    
#    
#    '''
#    1) build the model identifying specific reviews
#    - option 1: miss this step
#    - option 2: active learning
#    '''
#    
#    infile = path+'reviews_features.json'
#    outfile = path+'specific_reviews_features.json'
    
#    #option 1
#    import shutil
#    shutil.copyfile(infile,outfile)
    
#    #option 2
#    message1 = None
#    message2 = None
##    message1 = activeLearning(NAME, path, infile, iterations = 10, portion = 10)
#    message2 = apply_model(NAME, path, infile, outfile)
#    
#    outfile = open(path+'results/activeLearning.txt','a')
#    outfile.write('\n===\n')
#    if message1:
#        outfile.write(message1)
#    if message2:
#        outfile.write(message2)
#    outfile.close()

    
    
    '''
    2) sample TRAIN ( STAT / TRAIN ) / TEST 
    '''
    infile = path+'specific_reviews_features.json'
#    sampler(path, infile, [0.4,0.8], busThres = 4, userThres = 1)
    
    
    
    '''
    3) build user and business profiles
    identify critical, frequent and important features
    (importantFeaturesIdentification.py)
    '''
    
#    importantFeatureIdentification(path+'specific_reviews_stat.json',
#                                   path+'businessProfile.json',
#                                   path+'userProfile.json',
#                                   True,True,1000000000)
    
    '''
    4) learn existence model
    (run_learnFeatureExistence.py)
    '''
     
#    numberFE = learnFE(path, 2000000000)
    
#    numberFE = 26255
    numberFE = 582
#    numberFE = 43770
    
    
    '''
    5) applyFeatureExistance.py
    '''
#    modelfile = path + 'models/modelDict_%d.model'%numberFE
    
#    applyFE(path, modelfile, 20000000000)
    
    
    
    '''
    6) computeStat.py
    '''
    
#    computeStat(path, modelfile, 20000000000)
    
    
    
    
    '''
    7) Aspect selection
    aspectStat.py
    '''
    
#    aspectStat(path)
    
    
    '''
    8) predict all features and sentiments
    pairCompare.py
    '''
    
#    predictAll(path, modelfile)
    
    '''
    9) pair compare
    pairCompare.py
    '''
#    pairCompare(path, modelfile)


    '''
    10) MF part
    learnSentimentMF.py
    '''
#    from sentimentPrediction.learnSentimentMF import learnSentimentMF
#    from sentimentPrediction.applySentimentMF import applySMF
#    learnSentimentMF(path,200000000)
#    applySMF(path, 500000000)

    
    '''
    11) aspect dependence
    dependence.py
    '''
    infile = path+'specific_reviews_features.json'
    outIndividual = path+'/dependence/aspect_stars.csv'
    outPairs = path+'/dependence/aspects_relations.csv'
    aspDep = aspectDependence()
    aspDep.readReviewsFromFile(infile)
    aspDep.computeDependence()
    aspDep.saveDependence(outIndividual,outPairs)
    
    
#    '''
#    12) item/user aspect dependence
#    dependence.py
#    '''
#    reviewFile = path+'specific_reviews_features.json'
#    userItemStats(path, reviewFile, minReviewUser = 10, minReviewItem = 50)