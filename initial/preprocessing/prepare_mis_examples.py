from textblob import TextBlob
from textblob import Word
import json
import re

from getKey import getKey

'''
Prepare examples of aspect/sentiment mistakes
Divide reviews to sentences and make POS (part of speech) tagging
Script: prepare_mis_examples.py
Input: 
   * ../../data/mistakes/BS_mistakes_with_numbers.txt
   * ../../data/mistakes/REST_mistakes_with_numbers.txt

Details: 
import TextBlob

Output: Build a text file with sentences and POS
Output format: sentenceID word/lemma (if lemma!=word)/POS
'''



train_files = [
               '../../data/mistakes/BS_mistakes_with_numbers.txt',
               '../../data/mistakes/REST_mistakes_with_numbers.txt']



def processSentence(sentence):
    result_list = list()
    wiki = TextBlob(sentence)
    for word_tuple in wiki.tags:
#        print(word_tuple)
        try:
            w = Word(word_tuple[0])
            if word_tuple[1].startswith('JJ'):
                k = 'a'
            else:
                k = word_tuple[1][0].lower()
            if k in {'c','p','i','w','d','t','m','e','u','f','s'}:
                norm = w.lemmatize()
            else:
                norm = w.lemmatize(k)
#            print(k,norm)
        except:
            #print word_tuple
            norm = ''

        if norm == word_tuple[0]:
            norm = ''
        result_list.append(word_tuple[0]+'/'+norm+'/'+word_tuple[1])
    return ' '.join(result_list)


# def processText(text, textid):
#     result_list = list()
#     zen = TextBlob(text)
#     senid = textid*10000
#     for sentence in zen.sentences:
#         result_list.append(processSentence(str(senid)+' '+sentence.raw))
#         senid += 1
#     del zen  
#     return result_list


def preProcessMistakes(filename,outfilename):
    infile = open(filename,"r")
    outfile = open(outfilename,"w")
    
    for line in infile:
        l = line.strip().split(' ',1)
        print(l)
        outfile.write(l[0]+'\t'+processSentence(re.sub(r'[^\x00-\x7F]', '_', l[1]))+'\n')
    
    infile.close()
    outfile.close()

if __name__ == '__main__':
    for f in train_files:
        of = f.replace('numbers.','POS.')
        preProcessMistakes(f,of)