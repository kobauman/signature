import sys
sys.path.append('../')
import logging
import time

from featureWorkers.importantFeaturesIdentification import importantFeatureIdentification
from featureWorkers.learnFeaturesExistence import learnFE
from featureWorkers.applyFeaturesExistence import applyFE

from params.params import path

if __name__ == '__main__':
    logger = logging.getLogger('signature')

    logfile = '../../data/log/%d_importantFeaturesIdentification.log'%int(time.time())
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
    

    
    '''
    3) importantFeaturesIdentification.py
    '''
    importantFeatureIdentification(path+'yelp_reviews_features_stat.json',
                                   path+'businessFeaturesAggregation_train.json',
                                   path+'userFeaturesAggregation_train.json',
                                   True,True,1000000000)

    
    '''
    4) run_learnFeatureExistence.py
    '''
    numberFE = learnFE(path, 10000000000)
    
    
    '''
    5) run_applyFeatureExistance.py
    '''
    modelfile = path + '/models/modelDict_%d.model'%numberFE
    trainAveragesFile = path+'/models/trainAverages_%d.model'%numberFE
    
    applyFE(path, modelfile, trainAveragesFile, 20000000000)