import sys
sys.path.append('../')
import logging
import time

from experiment.sampler import sampler
from featureWorkers.importantFeaturesIdentification import importantFeatureIdentification
from featureWorkers.learnFeaturesExistence import learnFE
from sentimentPrediction.learnSentimentMF import learnSentimentMF
from sentimentAggregation.learnAggregationModel import learnAggreationModel
from featureWorkers.applyFeaturesExistence import applyFE
from sentimentPrediction.applySentimentMF import applySMF
from sentimentAggregation.applyAggregationModel import applyAM
from standartAlgorithms.learnMatrixFactorization import learnMF
from standartAlgorithms.applyMatrixFactorization import applyMF
from evaluation.evaluate import evaluate

if __name__ == '__main__':
    logger = logging.getLogger('signature')

    logfile = '../../data/log/%d_signature.log'%int(time.time())
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
    

    #path = '../../data/restaurants'
    #path = '../../data/beautyspa'
    path = '../../data/beautyspa_r'
    '''
    to check quality of Bing's scripts:
    combineSentenciesWithSentiments.py
    
    1) run_dataCombiner.py
    '''

    '''
    2) run_sampler.py
    '''
    sampler(path+'/yelp_reviews_features.json', [0.4,0.8], 4)
    '''
    3) importantFeaturesIdentification.py
    '''
    importantFeatureIdentification(path+'/yelp_reviews_features_stat.json',
                                   path+'/businessFeaturesAggregation_train.json',
                                   path+'/userFeaturesAggregation_train.json',
                                   True,True,1000000000)

    
    '''
    4) run_learnFeatureExistence.py
    '''
    numberFE = learnFE(path, 10000000000)
    
    exit()
    '''
    5) run_learnSentimentMF.py
    '''
    learnSentimentMF(path,2000000000)
    
    '''
    6) run_learnAggregationModel.py
    '''
    numberAM = learnAggreationModel(path,2000000000)
    
    '''
    7) run_applyFeatureExistance.py
    '''
    modelfile = path + '/models/modelDict_%d.model'%numberFE
    trainAveragesFile = path+'/models/trainAverages_%d.model'%numberFE
    
    applyFE(path, modelfile, trainAveragesFile, 20000000000)
    
    '''
    8) run_applySentimentMF.py
    '''
    applySMF(path, 50000)
    
    '''
    9) run_applyAggregationModel.py
    '''
    applyAM(path,numberAM,200000)
    
    '''
    10) run_learnMF.py
    '''
    learnMF(path, 20000)
    
    '''
    11) run_applyMF.py
    '''
    applyMF(path, 4255)
    
    '''
    12) run_evaluation.py
    '''
    evaluate(path)