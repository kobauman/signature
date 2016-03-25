#!/usr/bin/env python
# encoding: utf-8
"""
USSSALoader.py
"""
import os
import re
import urllib
from zipfile import ZipFile
import csv
import pickle
        
work_path = '../../data/genderPrediction/'

def getNameList():
    if not os.path.exists(work_path+'names.pickle'):
        print('names.pickle does not exist, generating')
        if not os.path.exists(work_path+'names.zip'):
            print('names.zip does not exist, downloading from github')
            downloadNames()
        else:
            print('names.zip exists, not downloading')
    
        print('Extracting names from names.zip')
        namesDict=extractNamesDict()
        
        maleNames=list()
        femaleNames=list()
        
        print('Sorting Names')
        for name in namesDict:
            counts=namesDict[name]
            tuple=(name,counts[0],counts[1])
            if counts[0]>counts[1]:
                maleNames.append(tuple)
            elif counts[1]>counts[0]:
                femaleNames.append(tuple)
        
        names=(maleNames,femaleNames)
        
        print('Saving names.pickle')
        fw=open(work_path+'names.pickle','wb')
        pickle.dump(names,fw,-1)
        fw.close()
        print('Saved names.pickle')
    else:
        print('names.pickle exists, loading data')
        f=open(work_path+'names.pickle','rb')
        names=pickle.load(f)
        print('names.pickle loaded')
        
    print('%d male names loaded, %d female names loaded'%(len(names[0]),len(names[1])))
    
    print(names[0][0])
    
    output = open(work_path+'names.csv','w')
    output.write('gender,name,last_letter,last_two,last_three,last_four,last_five,last_is_vowel,first_letter,first_two,length\n')
    
    result = list()
    for gender_names in names:
        for name in gender_names:
            if name[1] > name[2]:
                gender = 'M'
            else:
                gender = 'F'
                
            res = [gender,name[0]]
            res.append(name[0][-1])
            res.append(name[0][-2:])
            res.append(name[0][-3:])
            res.append(name[0][-4:])
            res.append(name[0][-5:])
            res.append(str(int(name[0][-1] in 'AEIOUY')))
            res.append(name[0][0])
            res.append(name[0][:2])
            res.append(str(len(name[0])))
            result.append(','.join(res))
    output.write('\n'.join(result))
    output.close()
    
    return names
    
def downloadNames():
    u = urllib2.urlopen('https://github.com/downloads/sholiday/genderPredictor/names.zip')
    localFile = open('names.zip', 'w')
    localFile.write(u.read())
    localFile.close()
    
def extractNamesDict():
    zf=ZipFile(work_path+'names.zip', 'r')
    filenames=zf.namelist()
    
    names=dict()
    genderMap={'M':0,'F':1}
    
    for filename in filenames:
        if filename[-4:] not in ['.txt','.csv']:
            print('\tPassed file: %s'%filename)
            continue
        file=zf.open(filename,'r')
        rows=csv.reader(file, delimiter=',')
        
        for row in rows:
            try:
                name=row[0].upper()
                gender=genderMap[row[1]]
                count=int(row[2])
            except:
                print('PASS')
                continue
            
            if not names.has_key(name):
                names[name]=[0,0]
            names[name][gender]=names[name][gender]+count
        file.close()
        print('\tImported %s'%filename)
    return names

if __name__ == "__main__":
    getNameList()