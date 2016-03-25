
import numpy as np


def encodeAspects1features(fsw, review_features, featureAvgSent):
    features_list = list()
    for aspect in fsw.featureIdicator:
        if fsw.featureIdicator[aspect] == True:
            if aspect in review_features:
                sentiment = np.average(review_features[aspect])
            else:
                sentiment = featureAvgSent[aspect]
            
            features_list.append(sentiment)
    return features_list



def encodeAspects2features(fsw, review_features):
    features_list = list()
    for aspect in fsw.featureIdicator:
        if fsw.featureIdicator[aspect] == True:
            if aspect in review_features:
                sentiment = np.average(review_features[aspect])
            else:
                sentiment = 0.0
                
            if sentiment < 0.0:
                features_list.append(-sentiment)
            else:
                features_list.append(0.0)
            
            if sentiment > 0.0:
                features_list.append(sentiment)
            else:
                features_list.append(0.0)
    return features_list


