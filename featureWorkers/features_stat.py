
#6) Find a set of features for work
#based on statistics
#(mostly Manual)
#   * overall frequency
#   * frequency among users
#   * frequency among restaurants
#   * combination

import sys
import json
import logging
import numpy as np
from scipy import stats
            
            
work_data_path = "../../data/yelp/"

def featureStat():
    logger = logging.getLogger('stat.featureStat')
    filename = work_data_path + "yelp_training_set_review_sentiment_features.json"
    
    features_stats = dict()
    
    try:
        infile = open(filename,"r")
        review_dict = json.loads(infile.read())
        infile.close()
        logger.info('Reviews loaded from: %s'%filename)
    except:
        logger.error('Can\'t open file %s'%filename)
        return
    
    #groupID -> (sum(ratings), num(ratings))
    group_rating_stat = dict()
    
    #count raw ratings distribution
    y0 = {0:[0.0]*4,1:[0.0]*4,2:[0.0]*4,3:[0.0]*4,4:[0.0]*4,5:[0.0]*4}
    y = dict()  
    
    for i, business_user in enumerate(review_dict.keys()):
        review = review_dict[business_user]    
        for category in review['clusters']:
            cluster = review['clusters'][category]
            
            category_stats[category] = category_stats.get(category, {})
            category_stats[category][cluster] = category_stats[category].get(cluster,[])
            category_stats[category][cluster].append(review['stars'])
            
            y[category] = y.get(category,{0:[0.0]*4,1:[0.0]*4,2:[0.0]*4,3:[0.0]*4,4:[0.0]*4,5:[0.0]*4})
            y[category][review['stars']][cluster] += 1
            y[category][0][cluster]+=1
        
        if not i%5000:
            logger.info(str(i)+" reviews loaded")
        #if i > 1000:
        #    break;
         
    result = list()
    for category in category_stats:
        temp_list = list()
        temp_pairs_list = list()
        all_num = float(sum([len(category_stats[category][x]) for x in category_stats[category]]))/100.
        for i, group_key in enumerate(category_stats[category].keys()):
            avg = np.average(np.array(category_stats[category][group_key]))
            std = np.std(np.array(category_stats[category][group_key]))
            group_len = len(category_stats[category][group_key])
            temp_list.append('%d -> avg = %.3f; std = %.3f (%d %.2f %%)'%(group_key,avg,std,group_len,group_len/all_num))
        
            for j in range(i+1, len(category_stats[category].keys())):
                group_key_2 = category_stats[category].keys()[j]
                dif = sig_dif(category_stats[category][group_key],category_stats[category][group_key_2])
                temp_pairs_list.append('(%d,%d) difference -> %.3f'%(group_key,group_key_2,dif))
            
        logger.info('\n\n==================================\n'+category)
        logger.info('\n' + '\n'.join(temp_list))
        logger.info('\n' + '\n'.join(temp_pairs_list))
        
        result.append('\n\n==================================\n'+category)
        result.append('\n' + '\n'.join(temp_list))
        result.append('\n' + '\n'.join(temp_pairs_list))
        
        group_rates_list = list()
        for i in range(4):
            if y[category][0][i]:
                temp = [i,y[category][0][i], divide(y[category][1][i],y[category][0][i]),divide(y[category][2][i],y[category][0][i])]
                temp += [divide(y[category][3][i],y[category][0][i]),divide(y[category][4][i],y[category][0][i]),divide(y[category][5][i],y[category][0][i])]
                group_rates_list.append(' '.join([str(x) for x in temp]))
        
        logger.info('\n' + '\n'.join(group_rates_list))
        result.append('\n' + '\n'.join(group_rates_list))
    
    filename = work_data_path + "clusterStat.txt"
    outfile = open(filename,'w')
    outfile.write('\n'.join(result))
    outfile.close()
    
    
        

        
if __name__ == '__main__':
    logger = logging.getLogger('stat')

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    clusterStat()
    