import sys
sys.path.append('../')
import logging
import time

from featureWorkers.applyFeaturesExistence import applyFE

if __name__ == '__main__':
    logger = logging.getLogger('signature')

    logfile = '../../data/log/%d_applyFE.log'%int(time.time())
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
    

    path = '../../data/restaurants'
    modelfile = path + '/models/modelDict_102535_all.model'
    modelfile = path + '/models/modelDict_3001.model'
    applyFE(path, modelfile)