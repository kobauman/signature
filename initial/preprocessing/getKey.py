

def getKey(business, user, review, keyType = 'business_id'):
    if keyType == 'business_id':
        if not business:
            return
        return [(business['business_id']+"\t"+business['name']+"\t"+" ".join(business['categories'])).encode("utf8")]
    elif keyType == 'business_city':
        if not business:
            return
        return [business['city']]
    elif keyType == "categories":
        if not business:
            return
        #print business['categories']
        return business['categories']
    
    elif keyType == "restaurant":
        if not business:
            return
        if "Restaurants" in business['categories']:
            return 1
        else:
            return
        
    elif keyType == "Active Life":
        if not business:
            return
        if "Active Life" in business['categories']:
            return 1
        else:
            return
        
    elif keyType == "Shopping":
        if not business:
            return
        if "Shopping" in business['categories']:
            return 1
        else:
            return
        
    elif keyType == "Food":
        if not business:
            return
        if "Food" in business['categories']:
            return 1
        else:
            return
    
    elif keyType == "Nightlife":
        if not business:
            return
        if "Nightlife" in business['categories']:
            return 1
        else:
            return
    
    elif keyType == "BeautySpas":
        if not business:
            return
        if "Beauty & Spas" in business['categories']:
            return 1
        else:
            return
    
    elif keyType == "Automotive":
        if not business:
            return
        if "Automotive" in business['categories']:
            return 1
        else:
            return

    elif keyType == "hotels":
        if not business:
            return
        if "Hotels" in business['categories']:
            return 1
        else:
            return
        
    elif keyType == "BeautySpas":
        if not business:
            return
        if "Beauty & Spas" in business['categories']:
            return 1
        else:
            return
    
    elif keyType == "HotelRest":
        if not business:
            return
        if "Restaurants" in business['categories'] and "Hotels" in business['categories']:
            return 1
        else:
            return
    
    elif keyType == "italian":
        if not business:
            return
        if "Italian" in business['categories'] and "Restaurants" in business['categories']:
            return 1
        else:
            return
    
    elif keyType in  ['American (New)',
                      'Mexican','American (Traditional)',
                      'Pizza','Italian','Breakfast & Brunch',
                      'Sandwiches','Burgers','Sushi Bars',
                      'Chinese','Steakhouses','Japanese',
                      'Mediterranean','Asian Fusion','Thai','Seafood']:
        if not business:
            return
        if keyType in business['categories'] and "Restaurants" in business['categories']:
            return 1
        else:
            return   
    
    
    return -1

