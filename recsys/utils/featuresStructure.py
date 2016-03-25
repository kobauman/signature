import logging, sys,os





class featureStructureWorker:
    def __init__(self):
        self.feature_parents = dict()
        
        self.feature_parents['FOOD'] = []
        self.feature_parents['FOOD_FOOD'] = ['FOOD']
        self.feature_parents['FOOD_FOOD_MEAT'] = ['FOOD_FOOD','FOOD']
        
        self.feature_parents['FOOD_FOOD_MEAT_BEEF'] = ['FOOD_FOOD_MEAT','FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_MEAT_BURGER'] = ['FOOD_FOOD_MEAT','FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_MEAT_BACON'] = ['FOOD_FOOD_MEAT','FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_MEAT_RIB'] = ['FOOD_FOOD_MEAT','FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_MEAT_VEAL'] = ['FOOD_FOOD_MEAT','FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_MEAT_STEAK'] = ['FOOD_FOOD_MEAT','FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_MEAT_PORK'] = ['FOOD_FOOD_MEAT','FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_MEAT_LAMB'] = ['FOOD_FOOD_MEAT','FOOD_FOOD','FOOD']
        
        self.feature_parents['FOOD_FOOD_DISH'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_SIDE_VEGETABLES'] = ['FOOD_FOOD','FOOD','FOOD_FOOD_SIDE']
        self.feature_parents['FOOD_FOOD_BREAD'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_CHICKEN'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_DESSERT'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_SEAFOOD'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_SEAFOOD_SEA'] = ['FOOD_FOOD_SEAFOOD','FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_SEAFOOD_FISH'] = ['FOOD_FOOD_SEAFOOD','FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_CHEESE'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_SALAD'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_SIDE_POTATO'] = ['FOOD_FOOD','FOOD','FOOD_FOOD_SIDE']
        self.feature_parents['FOOD_FOOD_SOUP'] = ['FOOD_FOOD','FOOD']
        self.feature_parents['FOOD_FOOD_SAUCE'] = ['FOOD_FOOD','FOOD']
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
        self.feature_parents['DRINKS_NON-ALCOHOL'] = ['DRINKS']
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
        self.feature_parents['RESTAURANT_ENTERTAINMENT'] = ['RESTAURANT']
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

#        self.featureIdicator['FOOD'] = True
#        self.featureIdicator['FOOD_FOOD'] = True
#        self.featureIdicator['FOOD_FOOD_MEAT'] = True
#        
#        self.featureIdicator['FOOD_FOOD_MEAT_BEEF'] = True
#        self.featureIdicator['FOOD_FOOD_MEAT_BURGER'] = True
#        self.featureIdicator['FOOD_FOOD_MEAT_BACON'] = True
#        self.featureIdicator['FOOD_FOOD_MEAT_RIB'] = True
#        self.featureIdicator['FOOD_FOOD_MEAT_VEAL'] = True
#        self.featureIdicator['FOOD_FOOD_MEAT_STEAK'] = True
#        self.featureIdicator['FOOD_FOOD_MEAT_PORK'] = True
#        self.featureIdicator['FOOD_FOOD_MEAT_LAMB'] = True
#
#        self.featureIdicator['FOOD_FOOD_DISH'] = True
#        self.featureIdicator['FOOD_FOOD_SIDE_VEGETABLES'] = True
#        self.featureIdicator['FOOD_FOOD_BREAD'] = True
#        self.featureIdicator['FOOD_FOOD_CHICKEN'] = True
#        self.featureIdicator['FOOD_FOOD_DESSERT'] = True
#        self.featureIdicator['FOOD_FOOD_SEAFOOD'] = True
#        self.featureIdicator['FOOD_FOOD_SEAFOOD_SEA'] = True
#        self.featureIdicator['FOOD_FOOD_SEAFOOD_FISH'] = True
#
#        self.featureIdicator['FOOD_FOOD_CHEESE'] = True
#        self.featureIdicator['FOOD_FOOD_SALAD'] = True
#        self.featureIdicator['FOOD_FOOD_SIDE_POTATO'] = True
#        self.featureIdicator['FOOD_FOOD_SOUP'] = True
#        self.featureIdicator['FOOD_FOOD_SAUCE'] = True
#        self.featureIdicator['FOOD_FOOD_SIDE_PASTA'] = True
#        self.featureIdicator['FOOD_FOOD_SIDE_RICE'] = True
#        self.featureIdicator['FOOD_FOOD_SIDE'] = True
#        self.featureIdicator['FOOD_FOOD_SUSHI'] = True
#        self.featureIdicator['FOOD_FOOD_FRUIT'] = True
#        self.featureIdicator['FOOD_FOOD_EGGS'] = True
#        self.featureIdicator['FOOD_MEALTYPE_LUNCH'] = True
#        self.featureIdicator['FOOD_MEALTYPE_DINNER'] = True
#        self.featureIdicator['FOOD_MEALTYPE_MAIN'] = True
#        self.featureIdicator['FOOD_MEALTYPE_START'] = True
#        self.featureIdicator['FOOD_MEALTYPE_BRUNCH'] = True
#        self.featureIdicator['FOOD_MEALTYPE_BREAKFAST'] = True
#        self.featureIdicator['FOOD_SELECTION'] = True
#        self.featureIdicator['FOOD_PORTION'] = True
#        
#        self.featureIdicator['DRINKS'] = True
#        self.featureIdicator['DRINKS_ALCOHOL'] = True
#        self.featureIdicator['DRINKS_ALCOHOL_BEER'] = True
#        self.featureIdicator['DRINKS_ALCOHOL_WINE'] = True
#        self.featureIdicator['DRINKS_ALCOHOL_LIGHT'] = True
#        self.featureIdicator['DRINKS_ALCOHOL_HARD'] = True
#        self.featureIdicator['DRINKS_NON-ALCOHOL'] = True
#        self.featureIdicator['DRINKS_NON-ALCOHOL_COLD'] = True
#        self.featureIdicator['DRINKS_NON-ALCOHOL_HOT'] = True
#        
#        
#        self.featureIdicator['RESTAURANT'] = True
#        self.featureIdicator['RESTAURANT_ATMOSPHERE'] = True
#        self.featureIdicator['RESTAURANT_LOCATION'] = True
#        self.featureIdicator['RESTAURANT_CUSINE'] = True
#        self.featureIdicator['RESTAURANT_INTERNET'] = True
#        self.featureIdicator['RESTAURANT_INTERIOR'] = True
#        self.featureIdicator['RESTAURANT_MONEY'] = True
#        self.featureIdicator['RESTAURANT_PARKING'] = True
#        self.featureIdicator['RESTAURANT_ENTERTAINMENT'] = True
#        self.featureIdicator['RESTAURANT_ENTERTAINMENT_MUSIC'] = True
#        self.featureIdicator['RESTAURANT_ENTERTAINMENT_SPORT'] = True
#        
#        self.featureIdicator['SERVICE'] = True
#        
#        
#        self.featureIdicator['EXPERIENCE'] = True
#        self.featureIdicator['EXPERIENCE_TIME'] = True
#        self.featureIdicator['EXPERIENCE_COMPANY'] = True
#        self.featureIdicator['EXPERIENCE_RECOMMENDATIONS'] = True
#        self.featureIdicator['EXPERIENCE_OCCASION'] = True
#        self.featureIdicator['EXPERIENCE_BONUS'] = True
#        self.featureIdicator['EXPERIENCE_RESERVATION'] = True
#        self.featureIdicator['EXPERIENCE_TAKEOUT'] = True
#        
#        self.featureIdicator['PERSONAL'] = True
#        
#        self.featureIdicator['GENERAL'] = True
        
        
        
        
#        self.featureIdicator['0'] = True
#        self.featureIdicator['1'] = True
#        self.featureIdicator['2'] = True
#        self.featureIdicator['3'] = True
#        self.featureIdicator['4'] = True
#        self.featureIdicator['5'] = True
#        self.featureIdicator['6'] = True
#        self.featureIdicator['7'] = True
#        self.featureIdicator['8'] = True
#        self.featureIdicator['9'] = True
#        self.featureIdicator['10'] = True
#        self.featureIdicator['11'] = True
#        self.featureIdicator['12'] = True
#        self.featureIdicator['13'] = True
#        self.featureIdicator['14'] = True
#        self.featureIdicator['15'] = True
#        self.featureIdicator['16'] = True
#        self.featureIdicator['17'] = True
#        self.featureIdicator['18'] = True
#        self.featureIdicator['19'] = True
        
        '''TEST 32''' 
        self.featureIdicator['FOOD'] = True
        self.featureIdicator['FOOD_FOOD'] = True
        self.featureIdicator['FOOD_FOOD_MEAT'] = True
        
        self.featureIdicator['FOOD_FOOD_MEAT_BEEF'] = False
        self.featureIdicator['FOOD_FOOD_MEAT_BURGER'] = True
        self.featureIdicator['FOOD_FOOD_MEAT_BACON'] = False
        self.featureIdicator['FOOD_FOOD_MEAT_RIB'] = False
        self.featureIdicator['FOOD_FOOD_MEAT_VEAL'] = False
        self.featureIdicator['FOOD_FOOD_MEAT_STEAK'] = True #False
        self.featureIdicator['FOOD_FOOD_MEAT_PORK'] = False
        self.featureIdicator['FOOD_FOOD_MEAT_LAMB'] = False

        self.featureIdicator['FOOD_FOOD_DISH'] = True
        self.featureIdicator['FOOD_FOOD_SIDE_VEGETABLES'] = True
        self.featureIdicator['FOOD_FOOD_BREAD'] = True
        self.featureIdicator['FOOD_FOOD_CHICKEN'] = True
        self.featureIdicator['FOOD_FOOD_DESSERT'] = True
        self.featureIdicator['FOOD_FOOD_SEAFOOD'] = True
        self.featureIdicator['FOOD_FOOD_SEAFOOD_SEA'] = False
        self.featureIdicator['FOOD_FOOD_SEAFOOD_FISH'] = False

        self.featureIdicator['FOOD_FOOD_CHEESE'] = True
        self.featureIdicator['FOOD_FOOD_SALAD'] = True
        self.featureIdicator['FOOD_FOOD_SIDE_POTATO'] = True
        self.featureIdicator['FOOD_FOOD_SOUP'] = False
        self.featureIdicator['FOOD_FOOD_SAUCE'] = True
        self.featureIdicator['FOOD_FOOD_SIDE_PASTA'] = False
        self.featureIdicator['FOOD_FOOD_SIDE_RICE'] = False
        self.featureIdicator['FOOD_FOOD_SIDE'] = True
        self.featureIdicator['FOOD_FOOD_SUSHI'] = True
        self.featureIdicator['FOOD_FOOD_FRUIT'] = False
        self.featureIdicator['FOOD_FOOD_EGGS'] = True
        self.featureIdicator['FOOD_MEALTYPE_LUNCH'] = True #False
        self.featureIdicator['FOOD_MEALTYPE_DINNER'] = True
        self.featureIdicator['FOOD_MEALTYPE_MAIN'] = True
        self.featureIdicator['FOOD_MEALTYPE_START'] = False
        self.featureIdicator['FOOD_MEALTYPE_BRUNCH'] = False
        self.featureIdicator['FOOD_MEALTYPE_BREAKFAST'] = False
        self.featureIdicator['FOOD_SELECTION'] = True
        self.featureIdicator['FOOD_PORTION'] = False
        
        self.featureIdicator['DRINKS'] = True #True
        self.featureIdicator['DRINKS_ALCOHOL'] = True
        self.featureIdicator['DRINKS_ALCOHOL_BEER'] = True
        self.featureIdicator['DRINKS_ALCOHOL_WINE'] = False
        self.featureIdicator['DRINKS_ALCOHOL_LIGHT'] = False
        self.featureIdicator['DRINKS_ALCOHOL_HARD'] = False
        self.featureIdicator['DRINKS_NON-ALCOHOL'] = True
        self.featureIdicator['DRINKS_NON-ALCOHOL_COLD'] = False
        self.featureIdicator['DRINKS_NON-ALCOHOL_HOT'] = False
        
        
        self.featureIdicator['RESTAURANT'] = False #True
        self.featureIdicator['RESTAURANT_ATMOSPHERE'] = True #False
        self.featureIdicator['RESTAURANT_LOCATION'] = True
        self.featureIdicator['RESTAURANT_CUSINE'] = True
        self.featureIdicator['RESTAURANT_INTERNET'] = False
        self.featureIdicator['RESTAURANT_INTERIOR'] = True
        self.featureIdicator['RESTAURANT_MONEY'] = True
        self.featureIdicator['RESTAURANT_PARKING'] = False
        self.featureIdicator['RESTAURANT_ENTERTAINMENT'] = False
        self.featureIdicator['RESTAURANT_ENTERTAINMENT_MUSIC'] = False
        self.featureIdicator['RESTAURANT_ENTERTAINMENT_SPORT'] = False
        
        self.featureIdicator['SERVICE'] = True
        
        
        self.featureIdicator['EXPERIENCE'] = True
        self.featureIdicator['EXPERIENCE_TIME'] = True
        self.featureIdicator['EXPERIENCE_COMPANY'] = True
        self.featureIdicator['EXPERIENCE_RECOMMENDATIONS'] = False
        self.featureIdicator['EXPERIENCE_OCCASION'] = False
        self.featureIdicator['EXPERIENCE_BONUS'] = False
        self.featureIdicator['EXPERIENCE_RESERVATION'] = False
        self.featureIdicator['EXPERIENCE_TAKEOUT'] = False
        
        self.featureIdicator['PERSONAL'] = False
        
        self.featureIdicator['GENERAL'] = False

#===============================================================================================
#        '''TEST'''
#        self.featureIdicator['FOOD'] = False #True
#        self.featureIdicator['FOOD_FOOD'] = False #True
#        self.featureIdicator['FOOD_FOOD_MEAT'] = False #True
#        
#        self.featureIdicator['FOOD_FOOD_MEAT_BEEF'] = False
#        self.featureIdicator['FOOD_FOOD_MEAT_BURGER'] = False #True
#        self.featureIdicator['FOOD_FOOD_MEAT_BACON'] = False
#        self.featureIdicator['FOOD_FOOD_MEAT_RIB'] = False
#        self.featureIdicator['FOOD_FOOD_MEAT_VEAL'] = False
#        self.featureIdicator['FOOD_FOOD_MEAT_STEAK'] = False
#        self.featureIdicator['FOOD_FOOD_MEAT_PORK'] = False
#        self.featureIdicator['FOOD_FOOD_MEAT_LAMB'] = False
#
#        self.featureIdicator['FOOD_FOOD_DISH'] = False #True
#        self.featureIdicator['FOOD_FOOD_SIDE_VEGETABLES'] = False #True
#        self.featureIdicator['FOOD_FOOD_BREAD'] = False #True
#        self.featureIdicator['FOOD_FOOD_CHICKEN'] = False #True
#        self.featureIdicator['FOOD_FOOD_DESSERT'] = False #True
#        self.featureIdicator['FOOD_FOOD_SEAFOOD'] = False #True
#        self.featureIdicator['FOOD_FOOD_SEAFOOD_SEA'] = False
#        self.featureIdicator['FOOD_FOOD_SEAFOOD_FISH'] = False #True
#
#        self.featureIdicator['FOOD_FOOD_CHEESE'] = False #True
#        self.featureIdicator['FOOD_FOOD_SALAD'] = False #True
#        self.featureIdicator['FOOD_FOOD_SIDE_POTATO'] = False
#        self.featureIdicator['FOOD_FOOD_SOUP'] = False
#        self.featureIdicator['FOOD_FOOD_SAUCE'] = False #True
#        self.featureIdicator['FOOD_FOOD_SIDE_PASTA'] = False
#        self.featureIdicator['FOOD_FOOD_SIDE_RICE'] = False
#        self.featureIdicator['FOOD_FOOD_SIDE'] = False #True
#        self.featureIdicator['FOOD_FOOD_SUSHI'] = False #True
#        self.featureIdicator['FOOD_FOOD_FRUIT'] = False
#        self.featureIdicator['FOOD_FOOD_EGGS'] = False #True
#        self.featureIdicator['FOOD_MEALTYPE_LUNCH'] = False
#        self.featureIdicator['FOOD_MEALTYPE_DINNER'] = False #True
#        self.featureIdicator['FOOD_MEALTYPE_MAIN'] = False #True
#        self.featureIdicator['FOOD_MEALTYPE_START'] = False
#        self.featureIdicator['FOOD_MEALTYPE_BRUNCH'] = False
#        self.featureIdicator['FOOD_MEALTYPE_BREAKFAST'] = False
#        self.featureIdicator['FOOD_SELECTION'] = False #True
#        self.featureIdicator['FOOD_PORTION'] = False
#        
#        self.featureIdicator['DRINKS'] = False #True
#        self.featureIdicator['DRINKS_ALCOHOL'] = False #True
#        self.featureIdicator['DRINKS_ALCOHOL_BEER'] = False #True
#        self.featureIdicator['DRINKS_ALCOHOL_WINE'] = False
#        self.featureIdicator['DRINKS_ALCOHOL_LIGHT'] = False
#        self.featureIdicator['DRINKS_ALCOHOL_HARD'] = False
#        self.featureIdicator['DRINKS_NON-ALCOHOL'] = False #True
#        self.featureIdicator['DRINKS_NON-ALCOHOL_COLD'] = False
#        self.featureIdicator['DRINKS_NON-ALCOHOL_HOT'] = False
#        
#        
#        self.featureIdicator['RESTAURANT'] = False #True
#        self.featureIdicator['RESTAURANT_ATMOSPHERE'] = False #True
#        self.featureIdicator['RESTAURANT_LOCATION'] = False
#        self.featureIdicator['RESTAURANT_CUSINE'] = False
#        self.featureIdicator['RESTAURANT_INTERNET'] = False
#        self.featureIdicator['RESTAURANT_INTERIOR'] = False #True
#        self.featureIdicator['RESTAURANT_MONEY'] = True
#        self.featureIdicator['RESTAURANT_PARKING'] = False
#        self.featureIdicator['RESTAURANT_ENTERTAINMENT'] = False
#        self.featureIdicator['RESTAURANT_ENTERTAINMENT_MUSIC'] = False
#        self.featureIdicator['RESTAURANT_ENTERTAINMENT_SPORT'] = False
#        
#        self.featureIdicator['SERVICE'] = False #True
#        
#        
#        self.featureIdicator['EXPERIENCE'] = False #True
#        self.featureIdicator['EXPERIENCE_TIME'] = False #True
#        self.featureIdicator['EXPERIENCE_COMPANY'] = False #True
#        self.featureIdicator['EXPERIENCE_RECOMMENDATIONS'] = False
#        self.featureIdicator['EXPERIENCE_OCCASION'] = False
#        self.featureIdicator['EXPERIENCE_BONUS'] = False
#        self.featureIdicator['EXPERIENCE_RESERVATION'] = False
#        self.featureIdicator['EXPERIENCE_TAKEOUT'] = False
#        
#        self.featureIdicator['PERSONAL'] = False
#        
#        self.featureIdicator['GENERAL'] = False

#===============================================================================================
        
#        self.featureIdicator['FOOD'] = True
#        self.featureIdicator['FOOD_FOOD'] = True
#        self.featureIdicator['FOOD_FOOD_MEAT'] = True
#        
#        self.featureIdicator['FOOD_FOOD_MEAT_BEEF'] = False
#        self.featureIdicator['FOOD_FOOD_MEAT_BURGER'] = True
#        self.featureIdicator['FOOD_FOOD_MEAT_BACON'] = False
#        self.featureIdicator['FOOD_FOOD_MEAT_RIB'] = False
#        self.featureIdicator['FOOD_FOOD_MEAT_VEAL'] = False
#        self.featureIdicator['FOOD_FOOD_MEAT_STEAK'] = False
#        self.featureIdicator['FOOD_FOOD_MEAT_PORK'] = False
#        self.featureIdicator['FOOD_FOOD_MEAT_LAMB'] = False
#
#        self.featureIdicator['FOOD_FOOD_DISH'] = True
#        self.featureIdicator['FOOD_FOOD_SIDE_VEGETABLES'] = False
#        self.featureIdicator['FOOD_FOOD_BREAD'] = False
#        self.featureIdicator['FOOD_FOOD_CHICKEN'] = False
#        self.featureIdicator['FOOD_FOOD_DESSERT'] = False
#        self.featureIdicator['FOOD_FOOD_SEAFOOD'] = False
#        self.featureIdicator['FOOD_FOOD_SEAFOOD_SEA'] = False
#        self.featureIdicator['FOOD_FOOD_SEAFOOD_FISH'] = False
#
#        self.featureIdicator['FOOD_FOOD_CHEESE'] = False
#        self.featureIdicator['FOOD_FOOD_SALAD'] = False
#        self.featureIdicator['FOOD_FOOD_SIDE_POTATO'] = False
#        self.featureIdicator['FOOD_FOOD_SOUP'] = False
#        self.featureIdicator['FOOD_FOOD_SAUCE'] = True
#        self.featureIdicator['FOOD_FOOD_SIDE_PASTA'] = False
#        self.featureIdicator['FOOD_FOOD_SIDE_RICE'] = False
#        self.featureIdicator['FOOD_FOOD_SIDE'] = True
#        self.featureIdicator['FOOD_FOOD_SUSHI'] = True
#        self.featureIdicator['FOOD_FOOD_FRUIT'] = False
#        self.featureIdicator['FOOD_FOOD_EGGS'] = False
#        self.featureIdicator['FOOD_MEALTYPE_LUNCH'] = False
#        self.featureIdicator['FOOD_MEALTYPE_DINNER'] = False
#        self.featureIdicator['FOOD_MEALTYPE_MAIN'] = False
#        self.featureIdicator['FOOD_MEALTYPE_START'] = False
#        self.featureIdicator['FOOD_MEALTYPE_BRUNCH'] = False
#        self.featureIdicator['FOOD_MEALTYPE_BREAKFAST'] = True
#        self.featureIdicator['FOOD_SELECTION'] = True
#        self.featureIdicator['FOOD_PORTION'] = False
#        
#        self.featureIdicator['DRINKS'] = True
#        self.featureIdicator['DRINKS_ALCOHOL'] = True
#        self.featureIdicator['DRINKS_ALCOHOL_BEER'] = True
#        self.featureIdicator['DRINKS_ALCOHOL_WINE'] = False
#        self.featureIdicator['DRINKS_ALCOHOL_LIGHT'] = False
#        self.featureIdicator['DRINKS_ALCOHOL_HARD'] = False
#        self.featureIdicator['DRINKS_NON-ALCOHOL'] = False
#        self.featureIdicator['DRINKS_NON-ALCOHOL_COLD'] = False
#        self.featureIdicator['DRINKS_NON-ALCOHOL_HOT'] = False
#        
#        
#        self.featureIdicator['RESTAURANT'] = True
#        self.featureIdicator['RESTAURANT_ATMOSPHERE'] = False
#        self.featureIdicator['RESTAURANT_LOCATION'] = False
#        self.featureIdicator['RESTAURANT_CUSINE'] = False
#        self.featureIdicator['RESTAURANT_INTERNET'] = False
#        self.featureIdicator['RESTAURANT_INTERIOR'] = True
#        self.featureIdicator['RESTAURANT_MONEY'] = True
#        self.featureIdicator['RESTAURANT_PARKING'] = False
#        self.featureIdicator['RESTAURANT_ENTERTAINMENT'] = False
#        self.featureIdicator['RESTAURANT_ENTERTAINMENT_MUSIC'] = False
#        self.featureIdicator['RESTAURANT_ENTERTAINMENT_SPORT'] = False
#        
#        self.featureIdicator['SERVICE'] = True
#        
#        
#        self.featureIdicator['EXPERIENCE'] = True
#        self.featureIdicator['EXPERIENCE_TIME'] = False
#        self.featureIdicator['EXPERIENCE_COMPANY'] = False
#        self.featureIdicator['EXPERIENCE_RECOMMENDATIONS'] = False
#        self.featureIdicator['EXPERIENCE_OCCASION'] = False
#        self.featureIdicator['EXPERIENCE_BONUS'] = False
#        self.featureIdicator['EXPERIENCE_RESERVATION'] = False
#        self.featureIdicator['EXPERIENCE_TAKEOUT'] = False
#        
#        self.featureIdicator['PERSONAL'] = False
#        
#        self.featureIdicator['GENERAL'] = False
        
        
        
    
#    def __init__(self):
#        self.feature_parents = dict()
#        
#        
#        self.feature_parents['STAFF'] = []
#        self.feature_parents['STAFF_MASTER'] = ['STAFF']
#        self.feature_parents['STAFF_SERVICE'] = ['STAFF']
#        self.feature_parents['STAFF_DOCTOR'] = ['STAFF']
#        self.feature_parents['STAFF_INSTRUCTOR'] = ['STAFF']
#        self.feature_parents['STAFF_OWNER'] = ['STAFF']
#        
#        self.feature_parents['FOOD'] = []
#        self.feature_parents['FOOD_DRINK'] = ['FOOD']
#        self.feature_parents['FOOD_FOOD'] = ['FOOD']
#        
#        self.feature_parents['PROCEDURE'] = []
#        self.feature_parents['PROCEDURE_RELAX'] = ['PROCEDURE']
#        self.feature_parents['PROCEDURE_RELAX_MASSAGE'] = ['PROCEDURE_RELAX','PROCEDURE']
#        self.feature_parents['PROCEDURE_RELAX_SPA'] = ['PROCEDURE_RELAX','PROCEDURE']
#        self.feature_parents['PROCEDURE_RELAX_TRAIN'] = ['PROCEDURE_RELAX','PROCEDURE']
#        self.feature_parents['PROCEDURE_BEAUTY'] = ['PROCEDURE']
#        self.feature_parents['PROCEDURE_BEAUTY_HAIR'] = ['PROCEDURE_BEAUTY','PROCEDURE']
#        self.feature_parents['PROCEDURE_BEAUTY_FACE'] = ['PROCEDURE_BEAUTY','PROCEDURE']
#        self.feature_parents['PROCEDURE_BEAUTY_BARBER'] = ['PROCEDURE_BEAUTY','PROCEDURE']
#        self.feature_parents['PROCEDURE_BEAUTY_NAILS'] = ['PROCEDURE_BEAUTY','PROCEDURE']
#        self.feature_parents['PROCEDURE_BEAUTY_NAILS_MANI'] = ['PROCEDURE_BEAUTY_NAILS','PROCEDURE_BEAUTY','PROCEDURE']
#        self.feature_parents['PROCEDURE_BEAUTY_NAILS_PEDI'] = ['PROCEDURE_BEAUTY_NAILS','PROCEDURE_BEAUTY','PROCEDURE']
#        self.feature_parents['PROCEDURE_BEAUTY_WAX'] = ['PROCEDURE_BEAUTY','PROCEDURE']
#        
#        self.feature_parents['EXPERIENCE'] = []
#        self.feature_parents['EXPERIENCE_WAIT'] = ['EXPERIENCE']
#        self.feature_parents['EXPERIENCE_TIME'] = ['EXPERIENCE']
#        self.feature_parents['EXPERIENCE_COMPANY'] = ['EXPERIENCE']
#        self.feature_parents['EXPERIENCE_RECOMMENDATIONS'] = ['EXPERIENCE']
#        self.feature_parents['EXPERIENCE_OCCASION'] = ['EXPERIENCE']
#        self.feature_parents['EXPERIENCE_APPOINTMENT'] = ['EXPERIENCE']
#        self.feature_parents['EXPERIENCE_CONVERSATION'] = ['EXPERIENCE']
#        self.feature_parents['EXPERIENCE_FEELINGS'] = ['EXPERIENCE']
#                
#        self.feature_parents['SALON'] = []
#        self.feature_parents['SALON_LOCATION'] = ['SALON']
#        self.feature_parents['SALON_ATMOSPHERE'] = ['SALON']
#        self.feature_parents['SALON_TYPE'] = ['SALON']
#        self.feature_parents['SALON_INTERIOR'] = ['SALON']
#        self.feature_parents['SALON_INTERIOR_ROOM'] = ['SALON_INTERIOR','SALON']
#        self.feature_parents['SALON_INTERIOR_BATH'] = ['SALON_INTERIOR','SALON']
#        self.feature_parents['SALON_INTERIOR_DESIGN'] = ['SALON_INTERIOR','SALON']
#        self.feature_parents['SALON_PRICE'] = ['SALON']
#        self.feature_parents['SALON_PRICE_BONUS'] = ['SALON_PRICE','SALON']
#        self.feature_parents['SALON_PARKING'] = ['SALON']
#        self.feature_parents['SALON_ADDITIONAL'] = ['SALON']
#        self.feature_parents['SALON_EQUIPMENT'] = ['SALON']
#        
#        self.feature_parents['PERSONAL'] = []
#        
#        
#        
#        self.featureIdicator = dict()
#        
#        self.featureIdicator['STAFF'] = True
#        self.featureIdicator['STAFF_MASTER'] = True
#        self.featureIdicator['STAFF_SERVICE'] = True
#        self.featureIdicator['STAFF_DOCTOR'] = True
#        self.featureIdicator['STAFF_INSTRUCTOR'] = True
#        self.featureIdicator['STAFF_OWNER'] = True
#        
#        self.featureIdicator['FOOD'] = True
#        self.featureIdicator['FOOD_FOOD'] = True
#        self.featureIdicator['FOOD_DRINK'] = True
#        
#        self.featureIdicator['PROCEDURE'] = True
#        self.featureIdicator['PROCEDURE_RELAX'] = True
#        self.featureIdicator['PROCEDURE_RELAX_MASSAGE'] = True
#        self.featureIdicator['PROCEDURE_RELAX_SPA'] = True
#        self.featureIdicator['PROCEDURE_RELAX_TRAIN'] = True
#        self.featureIdicator['PROCEDURE_BEAUTY'] = True
#        self.featureIdicator['PROCEDURE_BEAUTY_HAIR'] = True
#        self.featureIdicator['PROCEDURE_BEAUTY_FACE'] = True
#        self.featureIdicator['PROCEDURE_BEAUTY_BARBER'] = True
#        self.featureIdicator['PROCEDURE_BEAUTY_NAILS'] = True
#        self.featureIdicator['PROCEDURE_BEAUTY_NAILS_MANI'] = True
#        self.featureIdicator['PROCEDURE_BEAUTY_NAILS_PEDI'] = True
#        self.featureIdicator['PROCEDURE_BEAUTY_WAX'] = True
#        
#        self.featureIdicator['EXPERIENCE'] = True
#        self.featureIdicator['EXPERIENCE_WAIT'] = True
#        self.featureIdicator['EXPERIENCE_TIME'] = True
#        self.featureIdicator['EXPERIENCE_COMPANY'] = True
#        self.featureIdicator['EXPERIENCE_RECOMMENDATIONS'] = True
#        self.featureIdicator['EXPERIENCE_OCCASION'] = True
#        self.featureIdicator['EXPERIENCE_APPOINTMENT'] = True
#        self.featureIdicator['EXPERIENCE_CONVERSATION'] = True
#        self.featureIdicator['EXPERIENCE_FEELINGS'] = True
#        
#        self.featureIdicator['SALON'] = True
#        self.featureIdicator['SALON_LOCATION'] = True
#        self.featureIdicator['SALON_ATMOSPHERE'] = True
#        self.featureIdicator['SALON_TYPE'] = True
#        self.featureIdicator['SALON_INTERIOR'] = True
#        self.featureIdicator['SALON_INTERIOR_ROOM'] = True
#        self.featureIdicator['SALON_INTERIOR_BATH'] = True
#        self.featureIdicator['SALON_INTERIOR_DESIGN'] = True
#        self.featureIdicator['SALON_PRICE'] = True
#        self.featureIdicator['SALON_PRICE_BONUS'] = True
#        self.featureIdicator['SALON_PARKING'] = True
#        self.featureIdicator['SALON_ADDITIONAL'] = True
#        self.featureIdicator['SALON_EQUIPMENT'] = True
#        
#        self.featureIdicator['PERSONAL'] = True
        
    
    def getReviewFeaturesExistence(self, review_features):
        result = dict()
        for sentence in review_features:
            for feature in review_features[sentence]:
                features_path = [feature]
                if feature in self.feature_parents:
                    features_path += self.feature_parents[feature]
                
                sentiment = float(review_features[sentence][feature])
                for f in features_path:
                    if f not in self.featureIdicator:
                        continue
                    if not self.featureIdicator[f]:
                        continue
                    result[f] = result.get(f,[])
                
                    #if sentiment:
                    result[f].append(sentiment)  
        return result
    
    def getReviewFeaturesSentiment(self, review_features):
        result = self.getReviewFeaturesExistence(review_features)
        #print result
        return {r:result[r] for r in result if len(result[r])>0}
    
    def getReviewFeatures(self, review_features):
        result = set()
        for sentence in review_features:
            for feature in review_features[sentence]:
                features_path = [feature]
                if feature in self.feature_parents:
                    features_path += self.feature_parents[feature]
                
                for f in features_path:
                    if f not in self.featureIdicator:
                        continue
                    if not self.featureIdicator[f]:
                        continue
                    result.add(f)
        aspects_dict = dict()
        for aspect in self.featureIdicator:
            if self.featureIdicator[aspect]:
                aspects_dict[aspect] = int(aspect in result)  
        return aspects_dict
    
            
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    