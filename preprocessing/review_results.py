import json

path  = '../../data/bing/American_New/'

sentence_dict = dict()
input1 = open(path + 'yelp_rest_revies_American _New_.txt', 'r')

for line in input1:
    l = [x.split('/')[0] for x in line.strip().split(' ')]
    sentence_dict[l[0]] = l[1:]
input1.close()
print 'Sentencies loaded.'
print sentence_dict['50000']


sentiment_dict = dict()
input2 = open(path + 'yelp_American_New-124_1.txt', 'r')
for line in input2:
    l = [x.strip() for x in line.strip().split('|')]
    #print l
    if l[1] == 'S':
        sentiment_dict[l[0]] = sentiment_dict.get(l[0],[])
        sentiment_dict[l[0]].append([int(x) for x in l[5:8]])
input2.close()
print 'Sentencies loaded.'
print sentiment_dict['50000']

output_file = open(path + 'American_New_compiled.txt', 'w')
for sentID in sentiment_dict:
    output_file.write('\n'+sentID+'\t'+' '.join(sentence_dict[sentID])+'\n')
    for info in sentiment_dict[sentID]:
        start = info[0]
        end = info[1]
        output_file.write(str(info)+' ->\t'+' '.join(sentence_dict[sentID][start-1:end])+'\n')
    
  
output_file.close()  

