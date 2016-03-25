import json
import os
import matplotlib.pyplot as plt


NAME = 'restaurants_a'
NAME = 'beautyspa'
path = '../../../data_recsys/' + NAME


b_file = path+'/businessProfile.json'
u_file = path+'/userProfile.json'
r_file = path+'/specific_reviews_extrain.json'
    
busImportantFeatures = json.loads(open(b_file,'r').readline())
userImportantFeatures = json.loads(open(u_file,'r').readline())


busHist = list()
for bus in busImportantFeatures:
    busHist.append(busImportantFeatures[bus]['reviewsNumber'])
    
userHist = list()
for user in userImportantFeatures:
    userHist.append(userImportantFeatures[user]['reviewsNumber'])

'''Write statistics and results'''
try:
    os.stat(path+'/stat/')
except:
    os.mkdir(path+'/stat/')
    
    
#fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
#ax0, ax1 = axes.flat
#
#ax0 = plt.figure()

n_bins = 19
plt.hist(busHist, n_bins, normed=0, alpha=0.9, label='',range = (0,20))
plt.
#ax0.legend(prop={'size': 10})
#ax0.set_title("Item reviews")
#print(type(ax0))
plt.suptitle('test title', fontsize=20)

#n_bins = 9
#ax1.hist(userHist, n_bins, normed=0, alpha=0.9, label='',range = (0,10))
##ax1.legend(prop={'size': 10})
##ax1.set_title("User reviews")

plt.tight_layout()
plt.savefig(path+'/stat/Review_dist.png')
    
    
