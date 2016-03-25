#!/usr/bin/env python
# encoding: utf-8
"""
sexPredictor.py
"""

from nltk import NaiveBayesClassifier,classify
import USSSALoader
import random
import pickle


work_path = '../../data/genderPrediction/'

#def nsyl(word):
#    lowercase = word.lower()
#    if lowercase not in d:
#        return -1
#    else:
#        return max([len([y for y in x if isdigit(y[-1])]) for x in d[lowercase]])

class genderPredictor():
    def save(self):
        pickle.dump(self.classifier,open(work_path+'sexModel.model','wb'))
        
    def load(self):
        self.classifier = pickle.load(open(work_path+'sexModel.model','r'))
    
    def getFeatures(self):
        maleNames,femaleNames=self._loadNames()
        
        featureset = list()
        for nameTuple in maleNames:
            features = self._nameFeatures(nameTuple[0])
            featureset.append((features,'M'))
        
        for nameTuple in femaleNames:
            features = self._nameFeatures(nameTuple[0])
            featureset.append((features,'F'))
    
        return featureset
    
    def trainAndTest(self,trainingPercent=0.80):
        featureset = self.getFeatures()
        random.shuffle(featureset)
        
        name_count = len(featureset)
        
        cut_point=int(name_count*trainingPercent)
        
        train_set = featureset[:cut_point]
        test_set  = featureset[cut_point:]
        
        self.train(train_set)
        self.save()
        return self.test(test_set)
        
    def classify(self,name):
        feats=self._nameFeatures(name)
        if self.classifier.classify(feats) == 'M':
            return 1
        else:
            return 0
        
    def train(self,train_set):
        self.classifier = NaiveBayesClassifier.train(train_set)
        return self.classifier
        
    def test(self,test_set):
        return classify.accuracy(self.classifier,test_set)
        
    def getMostInformativeFeatures(self,n=5):
        return self.classifier.most_informative_features(n)
        
    def _loadNames(self):
        return USSSALoader.getNameList()
        
    def _nameFeatures(self,name):
        name=name.upper()
        return {
            'last_letter': name[-1],
            'last_two' : name[-2:],
            'last_three' : name[-3:],
            'last_four' : name[-4:],
            'last_five' : name[-5:],
            'last_is_vowel' : (name[-1] in 'AEIOUY'),
            'first_letter' : name[0],
            'first_two' : name[:2],
            'length': len(name),
        }

if __name__ == "__main__":
    gp = genderPredictor()
    accuracy=gp.trainAndTest()
    print 'Accuracy: %f'%accuracy
    print 'Most Informative Features'
    feats=gp.getMostInformativeFeatures(10)
    for feat in feats:
        print '\t%s = %s'%feat
    
    print '\nStephen is classified as %d'%gp.classify('Stephen')
    print '\nMike is classified as %d'%gp.classify('Mike')
    names = ['Joe','Mike','Meghan','Teehee','Bill','Susana','Eric','Whitney','Beth and  James']
    for name in names:
        print '%s is classified as %d'%(name,gp.classify(name))