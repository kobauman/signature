import pandas as pd
import numpy as np
import json
import logging
import os

from utils.featuresStructure import featureStructureWorker

'''
ToDo:
+ logging
+ add stars dependence
+ add dependence orientation
- item/user information
- plot graph of aspects relationships
- get examples of sentiments
'''




def divide(x,y):
    if y == 0:
        return 100
    else:
        return (x / y)

class aspectDependence():
    
    def __init__(self):
        self.logger = logging.getLogger('signature.aspectDependence')
        self.logger.info('aspectDependence created')
    
        self.fsw = featureStructureWorker()
        
        self.aspectList = [x for x in self.fsw.featureIdicator if self.fsw.featureIdicator[x]]
        self.aspectList.sort()
        
        self.aspectStat = dict()
        for i,x in enumerate(self.aspectList):
            self.aspectStat[x] = pd.DataFrame(np.zeros((4,5)),
                                              index=['n','-1','0','1'],
                                              columns=[1,2,3,4,5])
        
        self.aspectPairStat = dict()
        for i,x in enumerate(self.aspectList):
            for y in self.aspectList[i+1:]:
                self.aspectPairStat[(x,y)] = pd.DataFrame(np.zeros((4,4)),
                                                          index=['n','-1','0','1'],
                                                          columns=['n','-1','0','1'])
        
        #resulting aspect-stars stat
        self.aspectStars = dict()
        #resulting dependence Stat
        self.resultingStat = dict()
        
        
    def readReviewsFromFile(self,infile):
        reviews = list()
        with open(infile,'r') as inputf:
            for i, line in enumerate(inputf):
                reviews.append(json.loads(line))
#                if i > 1000:
#                    break
        self.logger.info('%d reviews loaded'%len(reviews))
        self.readReviews(reviews)
        
#    @profile        
    def readReviews(self, reviewList):
        for i, review in enumerate(reviewList):
            '''get the set of review aspects with values'''
            reviewAspects = self.fsw.getReviewFeaturesSentiment(review['features'])
            if i%100 == 0:
                self.logger.debug('%d reviews parsed'%i)
#            if i > 1000:
#                break
            stars = review['stars']
            
            for i,x in enumerate(self.aspectList):
                if x in reviewAspects:
                    xv = str(int(np.sign(np.average(reviewAspects[x]))))
                else:
                    xv = 'n'
                
                #collect aspect Stat
                self.aspectStat[x][stars][xv] += 1
                
                for y in self.aspectList[i+1:]:
                    if y in reviewAspects:
                        yv = str(int(np.sign(np.average(reviewAspects[y]))))
                    else:
                        yv = 'n'
                    
                    #collect pairStat
                    self.aspectPairStat[(x,y)][xv][yv] += 1
        
    
    def computeDependence(self):
        '''
        For each pair of aspects compute the following numbers:
        - (a) number of reviews in intersection
        - (b) |intersection|/|Y|
        - (c) weight of Y depends on X (without "n")
        - (d) weight of Y depends on X (with "n")
        '''
        self.logger.info('Start computing stat for %d aspects'%len(self.aspectList))
        
        for i,x in enumerate(self.aspectList):
            #compute individual stat
            stat = self.aspectStat[x]
            existence = stat.loc[['-1','0','1']]
            if existence.sum().sum() < 10 :
                continue
            self.aspectStars[x] = {}
            self.aspectStars[x]['existence'] = existence.sum().sum()
            self.aspectStars[x]['% of reviews'] = existence.sum().sum()/stat.sum().sum()
            
            
            den = existence.sum().sum() - existence.T.max().sum()
            dev = existence.sum().sum() - existence.sum().max()
            self.aspectStars[x]['dependence'] = 1 - divide(den,dev)
            
#            if self.aspectStars[x]['dependence'] != 0:
#                print(x)
#                print(stat)
#                print(den,dev,self.aspectStars[x]['dependence'])
            
            d_sum = 0
            for x1, rating in enumerate([1,2,3,4,5]):
                for x2, sentiment in enumerate(['-1','0','1']):
                    d_sum += existence[rating][sentiment]*((x1-x2)**2)
            n = existence.sum().sum()
            self.aspectStars[x]['spearman'] = 1 - 6*d_sum/(n*(n**2-1)) 
                
            den = stat.sum().sum() - stat.T.max().sum()
            dev = stat.sum().sum() - stat.sum().max()
            self.aspectStars[x]['dependence with none'] = 1 - divide(den,dev)
            
            
            #compute pair stat
            for y in self.aspectList[i+1:]:
                if x==y:
                    continue
                pairStat = self.aspectPairStat[(x,y)]
                intersection = pairStat.loc[['-1','0','1']][['-1','0','1']]
                if intersection.sum().sum() < 10:
                    continue
                self.resultingStat[(x,y)] = {}
                self.resultingStat[(x,y)]['intersection'] = intersection.sum().sum()
                
                self.resultingStat[(x,y)]['% of intersection'] = divide(self.resultingStat[(x,y)]['intersection'],pairStat.loc[['-1','0','1']].sum().sum())
                
                den = intersection.sum().sum() - intersection.max().sum()
                dev = intersection.sum().sum() - intersection.T.sum().max()
                self.resultingStat[(x,y)]['dependence'] = 1 - divide(den,dev)
                
                d_sum = 0
                for x1, sent1 in enumerate(['-1','0','1']):
                    for x2, sent2 in enumerate(['-1','0','1']):
                        d_sum += intersection[sent1][sent2]*((x1-x2)**2)
                n = intersection.sum().sum()
                self.resultingStat[(x,y)]['spearman'] = 1 - 6*d_sum/(n*(n**2-1)) 
                
                den = pairStat.sum().sum() - pairStat.max().sum()
                dev = pairStat.sum().sum() - pairStat.T.sum().max()
                self.resultingStat[(x,y)]['dependence with none'] = 1 - divide(den,dev)
                
                
                #===============
                self.resultingStat[(y,x)] = {}
                pairStat = pairStat.T
                intersection = pairStat.loc[['-1','0','1']][['-1','0','1']]
                self.resultingStat[(y,x)]['intersection'] = intersection.sum().sum()
                
                self.resultingStat[(y,x)]['% of intersection'] = divide(self.resultingStat[(y,x)]['intersection'],pairStat.loc[['-1','0','1']].sum().sum())
                
                den = intersection.sum().sum() - intersection.max().sum()
                dev = intersection.sum().sum() - intersection.T.sum().max()
                self.resultingStat[(y,x)]['dependence'] = 1 - divide(den,dev)
                
                d_sum = 0
                for x1, sent1 in enumerate(['-1','0','1']):
                    for x2, sent2 in enumerate(['-1','0','1']):
                        d_sum += intersection[sent1][sent2]*((x1-x2)**2)
                n = intersection.sum().sum()
                self.resultingStat[(y,x)]['spearman'] = 1 - 6*d_sum/(n*(n**2-1))
                
                den = pairStat.sum().sum() - pairStat.max().sum()
                dev = pairStat.sum().sum() - pairStat.T.sum().max()
                self.resultingStat[(y,x)]['dependence with none'] = 1 - divide(den,dev)
            if i%10==0:
                self.logger.info('%d aspects processed'%i)
            
            
    def saveDependence(self, outIndividual = None, outPairs = None):
        #save individual aspect stat
        if outIndividual:
            self.aspectStars
            lines = list()
            for aspect in self.aspectStars:
                if self.aspectStars[aspect]['existence'] < 10:
                    continue
                lineList = [aspect]
                lineList.append(self.aspectStars[aspect]['existence'])
                lineList.append(self.aspectStars[aspect]['% of reviews'])
                lineList.append(self.aspectStars[aspect]['dependence'])
                lineList.append(self.aspectStars[aspect]['spearman'])
                lineList.append(self.aspectStars[aspect]['dependence with none'])
                lines.append(','.join([str(x) for x in lineList]))
                    
            with open(outIndividual,'w') as output:
                output.write(','.join(['aspect','existence','% of reviews',
                                       'dependence','spearman','dependence with none'])+'\n')
                output.write('\n'.join(lines))
        
        #save pair aspect stat
        if outPairs:
            lines = list()
            for pair in self.resultingStat:
                if self.resultingStat[pair]['intersection'] < 10:
                    continue
                lineList = list(pair)
                lineList.append(self.resultingStat[pair]['intersection'])
                lineList.append(self.resultingStat[pair]['% of intersection'])
                lineList.append(self.resultingStat[pair]['dependence'])
                lineList.append(self.resultingStat[pair]['spearman'])
                lineList.append(self.resultingStat[pair]['dependence with none'])
                lines.append(','.join([str(x) for x in lineList]))
                    
            with open(outPairs,'w') as output:
                output.write(','.join(['aspect1','aspect2','intersection','% of intersection',
                                       'dependence','spearman','dependence with none'])+'\n')
                output.write('\n'.join(lines))
            
            

def userItemStats(path, reviewFile, minReviewUser = 10, minReviewItem = 10):
    try:
        os.stat(path+'dependence/')
    except:
        os.mkdir(path+'dependence/')
    try:
        os.stat(path+'dependence/item')
    except:
        os.mkdir(path+'dependence/item')
    try:
        os.stat(path+'dependence/user')
    except:
        os.mkdir(path+'dependence/user')
       
    logger = logging.getLogger('signature.userItemStats')
    logger.info('userItemStats start')
    
    
    itemReviews = dict()
    userReviews = dict()
    with open(reviewFile,'r') as infile:
        for i, line in enumerate(infile):
            if i > 10000:
                break
            
            review = json.loads(line)
            busID = review['business_id']
            userID = review['user_id']
            
            itemReviews[busID] = itemReviews.get(busID,[])
            itemReviews[busID].append(review)
            userReviews[userID] = userReviews.get(userID,[])
            userReviews[userID].append(review)
    
    logger.info('%d reviews loaded for %d items and %d users'%(i,len(itemReviews),len(userReviews)))
    
    
    for busID in itemReviews:
        if len(itemReviews[busID]) >= minReviewItem:
            logger.debug('start work with %d reviews of item %s'%(len(itemReviews[busID]),busID))
            outIndividual = path+'/dependence/item/%d_%s_stars.csv'%(len(itemReviews[busID]),busID)
            outPairs = path+'/dependence/item/%d_%s_pairs.csv'%(len(itemReviews[busID]),busID)
            aspDep = aspectDependence()
            aspDep.readReviews(itemReviews[busID])
            aspDep.computeDependence()
            aspDep.saveDependence(outIndividual,outPairs)
    
    for userID in userReviews:
        if len(userReviews[userID]) >= minReviewUser:
            logger.debug('start work with %d reviews of user %s'%(len(userReviews[userID]),userID))
            outIndividual = path+'/dependence/user/%d_%s_stars.csv'%(len(userReviews[userID]),userID)
            outPairs = path+'/dependence/user/%d_%s_pairs.csv'%(len(userReviews[userID]),userID)
            aspDep = aspectDependence()
            aspDep.readReviews(userReviews[userID])
            aspDep.computeDependence()
            aspDep.saveDependence(outIndividual,outPairs)