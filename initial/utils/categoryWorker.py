#!/usr/bin/env python
# encoding: utf-8
"""
categoryWorker.py
"""

import json


work_path = '../../data/genderPrediction/'


class categoryWorker():
    def train(self):
        categories = set()
        train_data_path = "../../../yelp/data/splitting_reviews/"
        #load business information
        business_file = open(train_data_path + "yelp_training_set_business.json","r")
        for line in business_file:
            print len(line)
            l = json.loads(line)
            if 'Restaurants' in l['categories']:
                categories = categories.union(set(l['categories']))
        
        categories = list(categories)
        categories.sort()
        self.cat_dict = {cat:i for i,cat in enumerate(categories)}
        self.vec_len = len(self.cat_dict)
        
    def save(self):
        output = open(work_path+'categoryDict.json','w')
        output.write(json.dumps(self.cat_dict).encode('utf8', 'ignore'))
        output.close()
    
    def load(self):
        self.cat_dict = json.loads(open(work_path+'categoryDict.json','r').readline())
        self.vec_len = len(self.cat_dict)
        
    def classify(self,catSet):
        result = [0]*self.vec_len
        for cat in catSet:
            if cat in self.cat_dict:
                result[self.cat_dict[cat]] = 1
        return result
        
        

if __name__ == "__main__":
    cW = categoryWorker()
    cW.train()
    cW.save()
    print 'DONE'