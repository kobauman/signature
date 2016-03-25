import numpy as np  

def getFeatures(busID, userID, busImportantFeatures, userImportantFeatures, fsw):
    if busID not in busImportantFeatures or userID not in userImportantFeatures:
        return None
    #print busImportantFeatures[busID]['featureFreq']
    
    features = list()
    for aspect in fsw.featureIdicator:
        #print aspect
        if not fsw.featureIdicator[aspect]:
            continue
        bus_pfreq = busImportantFeatures[busID]['featureFreq'].get(aspect,0.0)
        #print aspect, bus_pfreq
        bus_tfidf = busImportantFeatures[busID]['tfidfDict'].get(aspect,0.0)
        bus_sent = busImportantFeatures[busID]['sentiment'].get(aspect,[0.0,0])[0]
        
        us_pfreq = userImportantFeatures[userID]['featureFreq'].get(aspect,0.0)
        us_tfidf = userImportantFeatures[userID]['tfidfDict'].get(aspect,0.0)
        us_sent = userImportantFeatures[userID]['sentiment'].get(aspect,[0.0,0])[0]
        
        features += [bus_pfreq,bus_tfidf,bus_sent,us_pfreq,us_tfidf,us_sent]
        
        features += [bus_pfreq*us_pfreq,bus_tfidf*us_tfidf]
        
        features += [bus_sent*us_pfreq,bus_sent*us_tfidf]
        
        
    busNum = np.log(1+busImportantFeatures[busID]['reviewsNumber'])
    busText = busImportantFeatures[busID]['textFeatures']
    
    userNum = np.log(1+userImportantFeatures[userID]['reviewsNumber'])
    userText = userImportantFeatures[userID]['textFeatures']
    
    features.append(busNum)
    features += busText
    
    features.append(userNum)
    features += userText
     
    return features
    
    