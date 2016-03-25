from textblob import TextBlob
# from textblob import Word
import json
import re

from getKey import getKey

import nltk
# from nltk.tag import pos_tag
# from nltk.tokenize import word_tokenize

from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()




'''
1) Divide reviews to sentences and make POS (part of speech) tagging
Script: pre_processing.py
Input: 
   * yelp_training_set_business.json
   * yelp_training_set_user.json
   * yelp_training_set_review.json
   * Category list

Details: 
sentenceID = reviewNumber*10000 + sentenceNumber
import TextBlob

Output: Build a text file with sentences and POS for each category
Output format: sentenceID word/lemma (if lemma!=word)/POS
'''



#train_data_path = "../../data/splitting_reviews/"

#train_data_path = "../../../context_discover/data/yelp_dataset_challenge_academic_dataset/"
train_data_path = "../../../data/YELP_DATASET/yelp_dataset_challenge_academic_dataset/"


#restKeys = ['American (New)','Mexican','American (Traditional)',
#            'Pizza','Italian','Breakfast & Brunch','Sandwiches',
#            'Burgers','Sushi Bars','Chinese','Steakhouses',
#            'Japanese','Mediterranean','Asian Fusion','Thai','Seafood']

restKeys = ['BeautySpas', 'restaurant', 'hotels']
# restKeys = ['BeautySpas']

#@profile
def processSentence(sentence):
    result_list = list()
    #wiki = TextBlob(sentence)
    
    tagset = None
    tokens = nltk.word_tokenize(sentence)
    tags = nltk.tag._pos_tag(tokens, tagset, tagger)
    
#     wiki = pos_tag(word_tokenize(sentence))
    for word_tuple in tags:
#        print(word_tuple)
        try:
#             w = Word(word_tuple[0])
            if word_tuple[1].startswith('JJ'):
                k = 'a'
            else:
                k = word_tuple[1][0].lower()
            if k in {'c','p','i','w','d','t','m','e','u','f','s'}:
                norm = lmtzr.lemmatize(word_tuple[0])
            else:
                norm = lmtzr.lemmatize(word_tuple[0],k)
#            print(k,norm)
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
    del zen  
    return result_list

#@profile
def preProcessReviews(limit = 100):
    #load business information
#    business_file = open(train_data_path + "yelp_training_set_business.json","r")
    business_file = open(train_data_path + "yelp_academic_dataset_business.json","r")
    
    business_dict = dict()
    for line in business_file:
        l = json.loads(line)
        if set(['Restaurants', 'Hotels', 'Beauty & Spas']).intersection(l['categories']):
#         if set(['Beauty & Spas']).intersection(l['categories']):
            business_dict[l['business_id']] = {'categories':l['categories']}
    print('Loaded %d businesses'%len(business_dict))
        
    #load user information
#    user_file = open(train_data_path + "yelp_training_set_user.json","r")
#     user_file = open(train_data_path + "yelp_academic_dataset_user.json","r")
#     user_dict = dict()
#     for line in user_file:
#         l = json.loads(line)
#         user_dict[l['user_id']] = l
#     print('Loaded %d users'%len(user_dict))
    
    #load reviews
#    review_file = open(train_data_path + "yelp_training_set_review.json","r")
    review_file = open(train_data_path + "yelp_academic_dataset_review.json","r")
    out_files = dict()
    cat_counter = dict()
    for keyType in restKeys:
        out_files[keyType] = open(train_data_path+'yelp_reviews_%s.txt'%(keyType.replace('(','_').replace(')','_').replace('&','_')),'w')
        cat_counter[keyType] = 0
        
    #out_file = open( + "yelp_training_set_bing_italian.txt","w")
    for counter, line in enumerate(review_file):
        if counter > limit:
            break
        review = json.loads(line)
        businessID = business_dict.get(review["business_id"],None);
        userID = None#user_dict.get(review["user_id"],None)
        
        for keyType in restKeys:
            curKeySet = getKey(businessID, userID, review, keyType)   
            if not curKeySet:
                continue
            elif curKeySet == -1:
                print('Wrong Key Type')
                exit() 
            cat_counter[keyType]+=1
            
            for sentence in processText(re.sub(r'[^\x00-\x7F]', '_', review['text']), counter):
                try:
                    out_files[keyType].write(sentence+'\n')
                except:
                    print(sentence)
        if counter%500 == 0:
            print('%d reviews processed'%counter)
            
    review_file.close()
    for keyType in restKeys:
        out_files[keyType].close()
    print(cat_counter)

if __name__ == '__main__':
    preProcessReviews(limit = 20000000000)
    #print processText('It gave good results. Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex.' , 42)