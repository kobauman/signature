import sys
sys.path.append('../')
import logging
import time

from featureWorkers.importantFeaturesIdentification import importantFeatureIdentification
from match.matchProfiles import matchProfiles

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
    importantFeatureIdentification(path+'yelp_reviews_features_train.json',
                                   path+'businessFeaturesAggregation_train.json',
                                   path+'userFeaturesAggregation_train.json',
                                   True,True,1000000000)

    
    '''
    12) run_learnFeatureExistence.py
    '''
    matchProfiles(path, 10000000000)