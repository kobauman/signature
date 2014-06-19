from textblob import TextBlob
from textblob import Word
import json

from getKey import getKey

train_data_path = "../../data/splitting_reviews/"

restKeys = ['American (New)','Mexican','American (Traditional)',
            'Pizza','Italian','Breakfast & Brunch','Sandwiches',
            'Burgers','Sushi Bars','Chinese','Steakhouses',
            'Japanese','Mediterranean','Asian Fusion','Thai','Seafood']

def processSentence(sentence):
    result_list = list()
    wiki = TextBlob(sentence)
    for word_tuple in wiki.tags:
        try:
            w = Word(word_tuple[0])
            if word_tuple[1].startswith('JJ'):
                k = 'a'
            else:
                k = word_tuple[1][0].lower()
            norm = w.lemmatize(k)
        except:
            #print word_tuple
            norm = ''

        if norm == word_tuple[0]:
            norm = ''
        result_list.append(word_tuple[0]+'/'+norm+'/'+word_tuple[1])
    return ' '.join(result_list)


def processText(text, textid):
    result_list = list()
    zen = TextBlob(text)
    senid = textid*10000
    for sentence in zen.sentences:
        result_list.append(processSentence(str(senid)+' '+sentence.raw))
        senid += 1
        
    return result_list


def preProcessReviews(limit = 100):
    #load business information
    business_file = open(train_data_path + "yelp_training_set_business.json","r")
    business_dict = dict()
    for line in business_file:
        l = json.loads(line)
        business_dict[l['business_id']] = l
    print 'Loaded %d businesses'%len(business_dict)
        
    #load user information
    user_file = open(train_data_path + "yelp_training_set_user.json","r")
    user_dict = dict()
    for line in user_file:
        l = json.loads(line)
        user_dict[l['user_id']] = l
    print 'Loaded %d users'%len(user_dict)
    
    #load reviews
    review_file = open(train_data_path + "yelp_training_set_review.json","r")
    out_files = dict()
    for keyType in restKeys:
        out_files[keyType] = open(train_data_path+'yelp_rest_revies_%s.txt'%(keyType.replace('(','_').replace(')','_').replace('&','_')),'w')
        
    #out_file = open( + "yelp_training_set_bing_italian.txt","w")
    for counter, line in enumerate(review_file):
        if counter > limit:
            break
        review = json.loads(line)
        businessID = business_dict.get(review["business_id"],None);
        userID = user_dict.get(review["user_id"],None)
        
        for keyType in restKeys:
            curKeySet = getKey(businessID, userID, review, keyType)   
            if not curKeySet:
                continue
            elif curKeySet == -1:
                print 'Wrong Key Type'
                exit() 
            
            for sentence in processText(review['text'], counter):
                out_files[keyType].write(sentence.encode('utf8', 'ignore')+'\n')
            #if counter%1 == 0:
            print '%d reviews processed'%counter
            
    review_file.close()
    for keyType in restKeys:
        out_files[keyType].close()


if __name__ == '__main__':
    preProcessReviews(limit = 200000000)
    #print processText('It gave good results. Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex.' , 42)