import json
import os
import matplotlib.pyplot as plt


NAME = 'restaurants'
#NAME = 'restaurants_a'
#NAME = 'beautyspa'
path = '../../../data_recsys/' + NAME


b_file = path+'/businessProfile.json'
u_file = path+'/userProfile.json'
r_file = path+'/specific_reviews_extrain.json'
    
busImportantFeatures = json.loads(open(b_file,'r').readline())
print('busImportantFeatures loaded')
userImportantFeatures = json.loads(open(u_file,'r').readline())
print('userImportantFeatures loaded')


'''Write statistics and results'''
try:
    os.stat(path+'/stat/')
except:
    os.mkdir(path+'/stat/')
    
#overall = path+'/specific_reviews_features.json'
train_file = open(path+'/specific_reviews_extrain.json', 'r')
test_file = open(path+'/specific_reviews_test.json', 'r')

busIDs = set()
userIDs = set()
reviews = 0
for i, line in enumerate(train_file):
    review = json.loads(line.strip())
    userID = review['user_id']
    busID = review['business_id']
    
    b = busImportantFeatures.get(busID,{'reviewsNumber':0})['reviewsNumber']
    u  = userImportantFeatures.get(userID,{'reviewsNumber':0})['reviewsNumber']
#    if  b > 0:
#        busIDs.add(busID)
#    if u > 0:
#        userIDs.add(userID)
    if b > 5 and u > 5:
        busIDs.add(busID)
        userIDs.add(userID)
        reviews += 1
    if not i%1000:
        print('%d reviews loaded'%i)
print(reviews, len(busIDs),len(userIDs))


busIDs = set()
userIDs = set()
reviews = 0
for i, line in enumerate(test_file):
    review = json.loads(line.strip())
    userID = review['user_id']
    busID = review['business_id']
    
    b = busImportantFeatures.get(busID,{'reviewsNumber':0})['reviewsNumber']
    u  = userImportantFeatures.get(userID,{'reviewsNumber':0})['reviewsNumber']
#    if  b > 0:
#        busIDs.add(busID)
#    if u > 0:
#        userIDs.add(userID)
    if b > 5 and u > 5:
        busIDs.add(busID)
        userIDs.add(userID)
        reviews += 1
#    print(i)
print(reviews, len(busIDs),len(userIDs))