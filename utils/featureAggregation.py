import sys
import copy
import numpy as np
sys.path.append('../')

from utils.featuresStructure import featureStructureWorker

#!!!NOT USED!!!

'''
get set of reviews with assigned features
return 
    dictionary: {feature: aggregated sentiment}
'''


def featureAggregation(review_list, ignore_neutral = True):
    fsw = featureStructureWorker()
    aggregation_dict = dict()
    for review in review_list:
        reviewFeatures = fsw.getReviewFeatures(review)
        for feature in reviewFeatures:
            aggregation_dict[feature] = aggregation_dict.get(feature,[])
            if ignore_neutral:
                aggregation_dict[feature].append(np.average([x for x in reviewFeatures[feature] if x]))
            else:
                aggregation_dict[feature].append(np.average(reviewFeatures[feature]))
    
    for feature in aggregation_dict:
        aggregation_dict[feature] = [round(np.average(aggregation_dict[feature]),3),
                                     len(aggregation_dict[feature])]
    
    return copy.deepcopy(aggregation_dict)