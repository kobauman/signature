import logging, sys,os





class featureStructureWorker:
    def __init__(self):
        self.feature_parents = dict()
        
        self.feature_parents['FOOD'] = []
        self.feature_parents['FOOD_FOOD'] = ['FOOD']
        self.feature_parents['FOOD_FOOD_SAUCE'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_MEAT'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_DISH'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_SIDE_VEGETABLES'] = ['FOOD_FOOD','FOOD','FOOD_FOOD_SIDE']
        self.feature_parents['FOOD_FOOD_BREAD'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_CHICKEN'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_DESSERT'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_SEAFOOD'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_CHEESE'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_SALAD'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_SIDE_POTATO'] = ['FOOD_FOOD','FOOD','FOOD_FOOD_SIDE']
        self.feature_parents['FOOD_FOOD_SOUP'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_SIDE_PASTA'] = ['FOOD_FOOD','FOOD','FOOD_FOOD_SIDE']
        self.feature_parents['FOOD_FOOD_SIDE_RICE'] = ['FOOD_FOOD','FOOD','FOOD_FOOD_SIDE']
        self.feature_parents['FOOD_FOOD_SIDE'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_SUSHI'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_FRUIT'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_EGGS'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_MEALTYPE_LUNCH'] = ['FOOD','FOOD_MEALTYPE']
        self.feature_parents['FOOD_MEALTYPE_DINNER'] = ['FOOD','FOOD_MEALTYPE']
        self.feature_parents['FOOD_MEALTYPE_MAIN'] = ['FOOD','FOOD_MEALTYPE']
        self.feature_parents['FOOD_MEALTYPE_START'] = ['FOOD','FOOD_MEALTYPE']
        self.feature_parents['FOOD_MEALTYPE_BRUNCH'] = ['FOOD','FOOD_MEALTYPE']
        self.feature_parents['FOOD_MEALTYPE_BREAKFAST'] = ['FOOD','FOOD_MEALTYPE']
        self.feature_parents['FOOD_SELECTION'] = ['FOOD']
        #self.feature_parents['FOOD_PORTION'] = ['FOOD']
        
        
    
        
        self.feature_parents['DRINKS'] = []
        self.feature_parents['DRINKS_ALCOHOL'] = ['DRINKS']
        self.feature_parents['DRINKS_ALCOHOL_BEER'] = ['DRINKS','DRINKS_ALCOHOL']
        self.feature_parents['DRINKS_ALCOHOL_WINE'] = ['DRINKS','DRINKS_ALCOHOL']
        self.feature_parents['DRINKS_ALCOHOL_LIGHT'] = ['DRINKS','DRINKS_ALCOHOL']
        self.feature_parents['DRINKS_ALCOHOL_HARD'] = ['DRINKS','DRINKS_ALCOHOL']
        self.feature_parents['DRINKS_NON-ALCOHOL_COLD'] = ['DRINKS','DRINKS_NON-ALCOHOL']
        self.feature_parents['DRINKS_NON-ALCOHOL_HOT'] = ['DRINKS','DRINKS_NON-ALCOHOL']
        
        
        self.feature_parents['RESTAURANT'] = []
        self.feature_parents['RESTAURANT_ATMOSPHERE'] = ['RESTAURANT']
        self.feature_parents['RESTAURANT_LOCATION'] = ['RESTAURANT']
        self.feature_parents['RESTAURANT_CUSINE'] = ['RESTAURANT']
        self.feature_parents['RESTAURANT_INTERNET'] = ['RESTAURANT']
        self.feature_parents['RESTAURANT_INTERIOR'] = ['RESTAURANT']
        self.feature_parents['RESTAURANT_MONEY'] = ['RESTAURANT']
        self.feature_parents['RESTAURANT_PARKING'] = ['RESTAURANT']
        self.feature_parents['RESTAURANT_ENTERTAINMENT_MUSIC'] = ['RESTAURANT','RESTAURANT_ENTERTAINMENT']
        self.feature_parents['RESTAURANT_ENTERTAINMENT_SPORT'] = ['RESTAURANT','RESTAURANT_ENTERTAINMENT']
        
        self.feature_parents['SERVICE'] = []
        
        
        self.feature_parents['EXPERIENCE'] = []
        self.feature_parents['EXPERIENCE_TIME'] = ['EXPERIENCE']
        self.feature_parents['EXPERIENCE_COMPANY'] = ['EXPERIENCE']
        self.feature_parents['EXPERIENCE_RECOMMENDATIONS'] = ['EXPERIENCE']
        self.feature_parents['EXPERIENCE_OCCASION'] = ['EXPERIENCE']
        self.feature_parents['EXPERIENCE_BONUS'] = ['EXPERIENCE']
        self.feature_parents['EXPERIENCE_RESERVATION'] = ['EXPERIENCE']
        self.feature_parents['EXPERIENCE_TAKEOUT'] = ['EXPERIENCE']
        
        self.feature_parents['PERSONAL'] = []
        
        self.feature_parents['GENERAL'] = []
        
        
        
        self.featureIdicator = dict()
        
        self.featureIdicator['FOOD'] = True
        self.featureIdicator['FOOD_FOOD'] = True
        self.featureIdicator['FOOD_FOOD_SAUCE'] = True
        self.featureIdicator['FOOD_FOOD_MEAT'] = True
        self.featureIdicator['FOOD_FOOD_DISH'] = True
        self.featureIdicator['FOOD_FOOD_SIDE_VEGETABLES'] = False
        self.featureIdicator['FOOD_FOOD_BREAD'] = True
        self.featureIdicator['FOOD_FOOD_CHICKEN'] = True
        self.featureIdicator['FOOD_FOOD_DESSERT'] = True
        self.featureIdicator['FOOD_FOOD_SEAFOOD'] = True
        self.featureIdicator['FOOD_FOOD_CHEESE'] = True
        self.featureIdicator['FOOD_FOOD_SALAD'] = True
        self.featureIdicator['FOOD_FOOD_SIDE_POTATO'] = False
        self.featureIdicator['FOOD_FOOD_SOUP'] = True
        self.featureIdicator['FOOD_FOOD_SIDE_PASTA'] = False
        self.featureIdicator['FOOD_FOOD_SIDE_RICE'] = False
        self.featureIdicator['FOOD_FOOD_SIDE'] = True
        self.featureIdicator['FOOD_FOOD_SUSHI'] = True
        self.featureIdicator['FOOD_FOOD_FRUIT'] = True
        self.featureIdicator['FOOD_FOOD_EGGS'] = True
        self.featureIdicator['FOOD_MEALTYPE_LUNCH'] = True
        self.featureIdicator['FOOD_MEALTYPE_DINNER'] = True
        self.featureIdicator['FOOD_MEALTYPE_MAIN'] = True
        self.featureIdicator['FOOD_MEALTYPE_START'] = True
        self.featureIdicator['FOOD_MEALTYPE_BRUNCH'] = True
        self.featureIdicator['FOOD_MEALTYPE_BREAKFAST'] = True
        self.featureIdicator['FOOD_SELECTION'] = True
        self.featureIdicator['FOOD_PORTION'] = True
        
        self.featureIdicator['DRINKS'] = True
        self.featureIdicator['DRINKS_ALCOHOL'] = True
        self.featureIdicator['DRINKS_ALCOHOL_BEER'] = True
        self.featureIdicator['DRINKS_ALCOHOL_WINE'] = True
        self.featureIdicator['DRINKS_ALCOHOL_LIGHT'] = True
        self.featureIdicator['DRINKS_ALCOHOL_HARD'] = True
        self.featureIdicator['DRINKS_NON-ALCOHOL_COLD'] = True
        self.featureIdicator['DRINKS_NON-ALCOHOL_HOT'] = True
        
        
        self.featureIdicator['RESTAURANT'] = True
        self.featureIdicator['RESTAURANT_ATMOSPHERE'] = True
        self.featureIdicator['RESTAURANT_LOCATION'] = True
        self.featureIdicator['RESTAURANT_CUSINE'] = True
        self.featureIdicator['RESTAURANT_INTERNET'] = True
        self.featureIdicator['RESTAURANT_INTERIOR'] = True
        self.featureIdicator['RESTAURANT_MONEY'] = True
        self.featureIdicator['RESTAURANT_PARKING'] = True
        self.featureIdicator['RESTAURANT_ENTERTAINMENT_MUSIC'] = True
        self.featureIdicator['RESTAURANT_ENTERTAINMENT_SPORT'] = True
        
        self.featureIdicator['SERVICE'] = True
        
        
        self.featureIdicator['EXPERIENCE'] = True
        self.featureIdicator['EXPERIENCE_TIME'] = True
        self.featureIdicator['EXPERIENCE_COMPANY'] = True
        self.featureIdicator['EXPERIENCE_RECOMMENDATIONS'] = True
        self.featureIdicator['EXPERIENCE_OCCASION'] = True
        self.featureIdicator['EXPERIENCE_BONUS'] = True
        self.featureIdicator['EXPERIENCE_RESERVATION'] = True
        self.featureIdicator['EXPERIENCE_TAKEOUT'] = True
        
        self.featureIdicator['PERSONAL'] = True
        
        self.featureIdicator['GENERAL'] = True
        
    
    def getReviewFeaturesExistence(self, review_features):
        result = dict()
        for sentence in review_features:
            for feature in review_features[sentence]:
                features_path = [feature]
                if feature in self.feature_parents:
                    features_path += self.feature_parents[feature]
                
                sentiment = int(review_features[sentence][feature])
                for f in features_path:
                    if f not in self.featureIdicator:
                        continue
                    if not self.featureIdicator[f]:
                        continue
                    result[f] = result.get(f,[])
                
                    if sentiment:
                        result[f].append(sentiment)  
        return result
    
    def getReviewFeaturesSentiment(self, review_features):
        result = self.getReviewFeaturesExistence(review_features)
        
        return {r:result[r] for r in result if len(result[r])>0}
    
    
            
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    