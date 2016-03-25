import json
import logging
import os
from scipy import stats
import numpy as np

import sys
sys.path.append('../')
from utils.featuresStructure import featureStructureWorker



def sig_dif(array1,array2):
    try:
        arr1 = np.array(array1)
        arr2 = np.array(array2)
        return round(stats.ttest_ind(arr1,arr2)[1],3)
    except:
        return None


def aspectStat(path):
    logger = logging.getLogger('signature.aspectStat')
    logger.info('start computing aspect Stat')
    #get data
    b_file = path+'/businessProfile.json'
    u_file = path+'/userProfile.json'
    
    busImportantFeatures = json.loads(open(b_file,'r').readline())
    logger.info('Important BUSINESS Features loaded')
    userImportantFeatures = json.loads(open(u_file,'r').readline())
    logger.info('Important USER Features loaded')

    
    aspectStat = dict()
    
    fsw = featureStructureWorker()
    for f, aspect in enumerate(fsw.featureIdicator):
        aspectStat[aspect] = {'total':0, 'bus10':0, 'user10':0, 'posNum':0, 'negNum':0,
                              'busDiff+-':0, 'userDiff+-':0, 'busDiff01':0, 'userDiff01':0}
        for busID in busImportantFeatures:
            bus_reviews = busImportantFeatures[busID]['reviewsNumber']
            bus_freq = busImportantFeatures[busID]['featureFreq'].get(aspect,0.0)
            aspectStat[aspect]['total'] += bus_freq/100.0 * bus_reviews
            if aspect in busImportantFeatures[busID]['critical']:
                aspectStat[aspect]['posNum'] += len(busImportantFeatures[busID]['critical'][aspect]['+'])
                aspectStat[aspect]['negNum'] += len(busImportantFeatures[busID]['critical'][aspect]['-'])
            
            if bus_freq > 10:
                aspectStat[aspect]['bus10'] += 1
                
                if aspect in busImportantFeatures[busID]['critical']:
                    exist = busImportantFeatures[busID]['critical'][aspect]['1']
                    pos = busImportantFeatures[busID]['critical'][aspect]['+']
                    neg = busImportantFeatures[busID]['critical'][aspect]['-']
#                    neutr = busImportantFeatures[busID]['critical'][aspect]['0']
                    none = busImportantFeatures[busID]['critical'][aspect]['n']
                    
                    if sig_dif(pos,neg) < 0.10501:
                        aspectStat[aspect]['busDiff+-'] += 1
                    
                    if sig_dif(exist,none) < 0.10501:
                        aspectStat[aspect]['busDiff01'] += 1
                        
        
        for userID in userImportantFeatures:
#            user_reviews = userImportantFeatures[userID]['reviewsNumber']
            user_freq = userImportantFeatures[userID]['featureFreq'].get(aspect,0.0)
            
            if user_freq > 1:
                aspectStat[aspect]['user10'] += 1
                
                if aspect in userImportantFeatures[userID]['critical']:
                    exist = userImportantFeatures[userID]['critical'][aspect]['1']
                    pos = userImportantFeatures[userID]['critical'][aspect]['+']
                    neg = userImportantFeatures[userID]['critical'][aspect]['-']
#                    neutr = userImportantFeatures[userID]['critical'][aspect]['0']
                    none = userImportantFeatures[userID]['critical'][aspect]['n']
                    
                    if sig_dif(pos,neg) < 0.10501:
                        aspectStat[aspect]['userDiff+-'] += 1
                    
                    if sig_dif(exist,none) < 0.10501:
                        aspectStat[aspect]['userDiff01'] += 1
                    
        logger.debug('done with (%d) %s'%(f,aspect))
    
    try:
        os.stat(path+'results/')
    except:
        os.mkdir(path+'results/')
    
    outfile = open(path+'/results/aspectStatistics.txt','w')
    outfile.write('total\tbus10\tuser10\tposNum\tnegNum\tbusDiff+-\tuserDiff+-\tbusDiff01\tuserDiff01\n')
    
    aspects = list(aspectStat.keys())
    aspects.sort()
    for aspect in aspects:
        r = aspectStat[aspect]
        outfile.write('%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n'%(aspect, r['total'], r['bus10'], r['user10'],
                                                                  r['posNum'],r['negNum'],
                                                                  r['busDiff+-'], r['userDiff+-'],
                                                                  r['busDiff01'], r['userDiff01']))
    outfile.close()