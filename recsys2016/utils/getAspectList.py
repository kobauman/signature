import logging
import re
import os


def getSetOfAspects(appName):
    logger = logging.getLogger('signature.getAspects')
    logger.info('starting getAspects')
    
    path = os.path.join('../../../data/','aspects', appName + '_aspects.txt')
    
    aspects = set()

    with open(path,'r') as af:
        for line in af:
            aspects.add(re.findall('([^\s]*)\)$',line)[-1].upper())
    aspects = list(aspects)
    aspects.sort()
#     print(len(aspects))
    return aspects




if __name__ == '__main__':
    
    appName = 'beautySpa'
    
    # 1) get list of features
    app_aspects = getSetOfAspects(appName)
    print(app_aspects)
    