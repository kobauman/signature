import json
import re, glob

#5) Combine all data together
#Input: sentenceID part_start part_end feature sentiment ([-1,0,1])
#Output: review -> {setence_num:{featureID:sentiment,..}, É}

#TODO combine data


path  = '../../data/'

def Filenames(directory):
    if not directory.endswith("/"):
            directory+="/";
    return glob.glob(directory+"*")
    
files = [x for x in Filenames(path) if 'output_' in x]
print files




for filename in files:
    infile = '_'.join(filename.replace('/output_','/').replace('yelp_','yelp_rest_revies_').split('_')[:-1])+'.txt'
    print infile

    sentence_dict = dict()
    input1 = open(infile, 'r')
    
    for line in input1:
        l = [x.split('/')[0] for x in line.strip().split(' ')]
        sentence_dict[l[0]] = l[1:]
    input1.close()
    print 'Sentencies loaded.'
    #print sentence_dict['50000']
    
    
    sentiment_dict = dict()
    input2 = open(filename, 'r')
    for line in input2:
        l = [x.strip() for x in line.strip().split('|')]
        #print l
        if l[1] == 'S':
            sentiment_dict[l[0]] = sentiment_dict.get(l[0],[])
            sentiment_dict[l[0]].append([int(x) for x in l[5:8]])
    input2.close()
    print 'Sentiments loaded.'
    #print sentiment_dict['50000']
    
    output_file = open(filename.replace('/output_','/compiled_'), 'w')
    for sentID in sentiment_dict:
        output_file.write('\n'+sentID+'\t'+' '.join(sentence_dict[sentID])+'\n')
        for info in sentiment_dict[sentID]:
            start = info[0]
            end = info[1]
            output_file.write(str(info)+' ->\t'+' '.join(sentence_dict[sentID][start-1:end])+'\n')
        
      
    output_file.close()  
