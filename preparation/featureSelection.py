import json
import re, glob
import sys
sys.path.append('../')

from utils.featuresStructure import featureStructureWorker
'''
6) Find a set of features for work
based on statistics
(mostly Manual)
   * overall frequency
   * frequency among users
   * frequency among restaurants
   * combination

Output: set of interesting features for work
'''


train_data_path = "../../../yelp/data/splitting_reviews/"

def featureStat(infileName, outfileName, limit = 1000):
    #load reviews
    review_file = open(infileName,"r")
    stat = [[],[],[]]
    feature_stat = dict()
    fsw = featureStructureWorker()
    for counter, line in enumerate(review_file):
        if counter > limit:
            break
        review = json.loads(line)
        #features = fsw.getFeatureAverage(review['features'])
        features = fsw.getReviewFeatures(review['features'])
        stat[0].append(review['ID'])
        stat[1].append(review['business_id'])
        stat[2].append(review['user_id'])
        for feature in features:
            feature_stat[feature] = feature_stat.get(feature,[[],[],[]])
            feature_stat[feature][0].append(review['ID'])
            feature_stat[feature][1].append(review['business_id'])
            feature_stat[feature][2].append(review['user_id'])
        if not counter %1000:
            print '%d reviews processed'%counter
        
        
    out_file = open(outfileName,"w")
    out_file.write('%30s\t%s\t%s\t%s\t%s\t%s\t%s\n'%('FeatureName','rewFreq','busFreq','usFreq',
                                         'rewPer','busPer','userPer'))
    
    result = []
    for i in range(len(stat)):
        stat[i] = float(len(set(stat[i])))/100
     
    for feature in feature_stat:
        for i in range(len(feature_stat[feature])):
            feature_stat[feature][i] = len(set(feature_stat[feature][i]))
        f = feature_stat[feature]
        s = '%30s\t%5d\t%5d\t%5d\t%5.1f\t%5.1f\t%5.1f\n'%(feature,f[0],f[1],f[2],
                                                    f[0]/stat[0],f[1]/stat[1],f[2]/stat[2])
        result.append([f[0]/stat[0],s])
        
    result.sort(reverse=True)
    for r in result:
        out_file.write(r[1])
            
    review_file.close()
    out_file.close()





if __name__ == '__main__':
    #path  = '../../data/'
    path = '../../data/bing/American_New'
    path = '../../data/restaurants'
    featureStat(path+'/yelp_reviews_features.json', path+'/feature_stat.txt', limit = 1000000)