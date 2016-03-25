import json
import logging
import pickle
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from scipy import stats
from scipy.interpolate import spline

from utils.featuresStructure import featureStructureWorker


def drawROC(y_true,y_pred_list, name, path):
    ind_list = [2,3,1,6]
    label_list = ['Random Forest', 'Random', 'Item Average', 'all', 'nothing', 'MF']
    color_cycle = ['r', 'g', 'b', 'y', 'm', 'purple']
#    label_list = ['Item Average', 'Random', 'Random Forest', 'all', 'nothing', 'MF']
#    color_cycle = ['b', 'g', 'r', 'y', 'm', 'purple']
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in ind_list:
        label = label_list[i-1]
        print(i,label)
        fpr[label], tpr[label], _ = roc_curve(y_true, y_pred_list[i-1])
        roc_auc[label] = auc(fpr[label], tpr[label])
    
#    ind_list = [2,1,3,6]
    plt.figure()
    for i in ind_list:
        l = label_list[i-1]
        plt.plot(fpr[l], tpr[l], linewidth=2.0, color = color_cycle[label_list.index(l)], label='%s (auc=%0.2f)' %(l,roc_auc[l]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curves')
    plt.legend(loc="lower right", prop={'size': 15})
    try:
        os.stat(path+'/testPictures/')
    except:
        os.mkdir(path+'/testPictures/')
    plt.savefig(path+'/testPictures/%s_ROC.png'%name)



def drawJaccDist(Jaccard, Jaccard_int, name, path):
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
#    ax0, ax1 = axes.flat
    
    ind_list = [2,3,1,6,4]
    label_list = ['Random', 'Item Average', 'Model', 'MF','all']
    
    ind_list = [3,1]
    label_list = ['Random Forest', 'Random', 'Item Average', 'all','ggg', 'MF']
    colors = ['r','r', 'b','b','r', 'purple']
    
    n_bins = 13
    for i in ind_list:
        y, binEdges, patches = ax0.hist(Jaccard[i], n_bins, normed=0,
                 alpha=0.0, color = colors[i-1],
                 #label=label_list[i-1],# range=(-0.0001, 1.0001),
                 histtype = 'step')
        
        x_sm = np.array(binEdges[:-1]+0.05)
        y_sm = np.array(y)
        x_smooth = np.linspace(x_sm.min(), x_sm.max(), 100)
        y_smooth = spline(x_sm, y_sm, x_smooth)
        
        ax0.plot(x_smooth, y_smooth, colors[i-1], linewidth=3,label=label_list[i-1])
        
        
        
#    ax0.hist([Jaccard[i] for i in ind_list], n_bins, normed=0, alpha=0.3, label=label_list, range=(-0.05, 1.05))
    #ax0.hist(compareList[1], n_bins, normed=1,alpha=0.5, histtype='bar', color='r', label='Loosers', range=(0.5, 5.5))
    ax0.legend(prop={'size': 25})
    #ax0.set_title('Jaccard Distribution %s'%name)
    
#    n_bins = 15
#    ax1.hist([Jaccard_int[i] for i in ind_list], n_bins, normed=0, alpha=0.5, histtype='bar', label=label_list, range=(-0.5, 15.5))
#    #ax0.hist(compareList[1], n_bins, normed=1,alpha=0.5, histtype='bar', color='r', label='Loosers', range=(0.5, 5.5))
#    ax1.legend(prop={'size': 10})
#    ax1.set_title('Intersection Distribution %s'%name)
    
    try:
        os.stat(path+'/testPictures/')
    except:
        os.mkdir(path+'/testPictures/')
    plt.savefig(path+'/testPictures/%s_Jdistr.png'%name)
    

def drawJacAcc(Jaccard, Accuracy, name, path):
    # Create plots with pre-defined labels.
    #fig = plt.figure()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax0, ax1 = axes.flat
    color_cycle=['r', 'g', 'b', 'y', 'm', 'purple']
    style_cycle = ['k','k','k--','k','k','k--']
    
    ax0.set_title('Jaccard %s'%name)
    for i in range(1,7):
        ax0.plot(Jaccard['th'], Jaccard[i], style_cycle[i-1], color = color_cycle[i-1], label='Pred %d'%(i))
    
    ax0.set_xticks(np.arange(0,1.,0.1))
    ax0.set_yticks(np.arange(0,1.,0.1))
    ax0.grid(True)
    ax0.legend(loc='lower left')
    
    ax1.set_title('Accuracy %s'%name)
    for i in range(1,7):
        ax1.plot(Accuracy['th'], Accuracy[i], style_cycle[i-1], color = color_cycle[i-1], label='Pred %d'%(i))
    ax1.set_xticks(np.arange(0,1.,0.1))
    ax1.set_yticks(np.arange(0,1.,0.1))
    ax1.grid(True)
    ax1.legend(loc='lower left')

    
    try:
        os.stat(path+'/testPictures/')
    except:
        os.mkdir(path+'/testPictures/')
    plt.savefig(path+'/testPictures/%s_Jaccard.png'%name)



def drawPR(y_true,y_pred_list,name,path,classes):
    precision = list()
    recall = list()
    thresholds = list()
    # get (precision, recall)
    
#    print(len(y_true),y_true[:20])
    
    labels = []
    for i, y_pred in enumerate(y_pred_list):
#        print('next',len(y_pred), y_pred[:20])
        try:
            pre, rec, thres = precision_recall_curve(y_true, y_pred)
            precision.append(pre)
            recall.append(rec)
            thresholds.append(thres)
            labels.append(i+1)
        except:
            pass
        
     
    thresholds = abs(classes[0]*np.ones(len(thresholds)) - thresholds)
    
      
    # Create plots with pre-defined labels.
    #fig = plt.figure()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    ax0, ax1, ax2, ax3 = axes.flat
    color_cycle=['r', 'g', 'b', 'y', 'm', 'purple']
    style_cycle = ['k','k','k--','k','k','k--']
    
    ax0.set_title('Precision %s'%name)
    for i in range(len(labels)):
        ax0.plot(thresholds[i], precision[i][:-1], style_cycle[i], color = color_cycle[i], label='Pred %d'%labels[i])
    
    ax0.set_xticks(np.arange(0,1.,0.1))
    ax0.set_yticks(np.arange(0,1.,0.1))
    ax0.grid(True)
    ax0.legend(loc='lower left')
    
    ax1.set_title('Recall %s'%name)
    for i in range(len(labels)):
        ax1.plot(thresholds[i], recall[i][:-1], style_cycle[i], color = color_cycle[i], label='Pred %d'%(i+1))
    ax1.set_xticks(np.arange(0,1.,0.1))
    ax1.set_yticks(np.arange(0,1.,0.1))
    ax1.grid(True)
    ax1.legend(loc='lower left')
    
    ax2.set_title('F1 %s'%name)
    for i in range(len(labels)):
        f1 = 2 *recall[i][:-1] * precision[i][:-1] / (recall[i][:-1] + precision[i][:-1])
        ax2.plot(thresholds[i], f1, style_cycle[i], color = color_cycle[i], label='Pred %d'%(i+1))
    ax2.set_xticks(np.arange(0,1.,0.1))
    ax2.set_yticks(np.arange(0,1.,0.1))
    ax2.grid(True)
    ax2.legend(loc='lower left')
    
    
    ax3.set_title('ROC %s'%name)
    for i in range(len(labels)):
        ax3.plot(recall[i][:-1], precision[i][:-1], style_cycle[i], color = color_cycle[i], label='Pred %d'%(i+1))
    ax3.set_xticks(np.arange(0,1.,0.1))
    ax3.set_yticks(np.arange(0,1.,0.1))
    ax3.grid(True)
    #ax3.legend(loc='lower left')
    
    try:
        os.stat(path+'/testPictures/')
    except:
        os.mkdir(path+'/testPictures/')
    plt.savefig(path+'/testPictures/%s.png'%name)




def computeStatWorker(testReviews, predType, path, modelDict, classes = [0, 1]):
    logger = logging.getLogger('signature.computeStat.cSW')
    logger.info('start computing Statistic from %d reviews for %s'%(len(testReviews), predType))
    fsw = featureStructureWorker()
    
    try:
        os.stat(path+'results/')
    except:
        os.mkdir(path+'results/')
    output = open(path+'results/example_%s_%d.txt'%(predType,classes[1]), 'w')
    
    
    Jaccard = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    Jaccard_int = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    
    Jaccard_vector = dict()# thres -> values
    Accuracy_vector = dict()
    
    Presision = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    Recall    = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    F1 = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    
    
    Presision_o = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
    Recall_o = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
    F1_o = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
    
    TP_o = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
    FP_o = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
    FN_o = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
    
    
    
    
#    RMSE = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
#    RMSE_o = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    
    
    aspectNumAvg = {'good':[], 0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    y_true = list()
    y_pred_list = [[],[],[],[],[],[]]
    
    for r, review in enumerate(testReviews):
        Jaccard_intersection = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
        Jaccard_union        = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
        
        Jaccard_vector_review = dict()
        Accuracy_vector_review = dict()
        for thres in np.arange(-0.05,1.05,0.05):
            Jaccard_vector_review[thres] = Jaccard_vector_review.get(thres, {1:[0,0], 2:[0,0], 3:[0,0], 4:[0,0],5:[0,0], 6:[0,0]})
            Accuracy_vector_review[thres] = Accuracy_vector_review.get(thres, {1:[0,0], 2:[0,0], 3:[0,0], 4:[0,0],5:[0,0], 6:[0,0]})
        
        
        TP = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
        FP = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
        FN = {1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
        
#        RMSE_review = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
        
        aspectNum = {'good':0.0, 0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0}
        if predType in review:
            aspectNum['good'] = len([aspect for aspect in review[predType] if fsw.featureIdicator[aspect]])
        

        if predType not in review:
            continue
#        print(review['exPredFeatures'])
        for feature in review[predType]:
            if not fsw.featureIdicator[feature]:
                continue
            
            for i in range(0,7):
                if abs(classes[1] - review[predType][feature][i]) < 0.5:
                    aspectNum[i] += 1
                
            
            #for plots
            y_true.append(int(review[predType][feature][0] == classes[1]))
            for i in range(1,7):
                y_pred_list[i-1].append(abs(classes[0] - review[predType][feature][i]))
                
            
            #for computing quality
            for i in range(1,7):
                realClass = review[predType][feature][0]
                if review[predType][feature][i] > 0.5:
                    predictedClass = 1
                else:
                    predictedClass = 0
                
                if realClass == classes[1]:
                    if predictedClass == classes[1]:
                        TP[i] += 1
                        TP_o[i] += 1
                    elif predictedClass == classes[0]:
                        FN[i] += 1
                        FN_o[i] += 1
                elif realClass == classes[0]:
                    if predictedClass == classes[1]:
                        FP[i] += 1
                        FP_o[i] += 1
                
                
                if realClass == classes[1] and predictedClass == classes[1]:
                    Jaccard_intersection[i] += 1
                
                if realClass == classes[1] or  predictedClass == classes[1]:
                    Jaccard_union[i] += 1
                
#                dif = pow(realClass - review[predType][feature][i], 2)
#                RMSE[i].append(dif)
#                RMSE_review[i].append(dif)
            
            '''
            Jaccard_vector
            '''
            for thres in np.arange(-0.05,1.05,0.05):
                for i in range(1,7):
                    if review[predType][feature][i] > thres:
                        predictedClass = 1
                    else:
                        predictedClass = 0
                    
                    if realClass == classes[1] and predictedClass == classes[1]:
                        Jaccard_vector_review[thres][i][0] += 1.0
                    
                    if realClass == classes[1] or  predictedClass == classes[1]:
                        Jaccard_vector_review[thres][i][1] += 1.0
                    
                    Accuracy_vector_review[thres][i][1] += 1.0    
                    if realClass == predictedClass:
                        Accuracy_vector_review[thres][i][0] += 1.0
            
            
        for i in range(1,7):
            if Jaccard_union[i]:
                Jaccard[i].append(Jaccard_intersection[i]/Jaccard_union[i])
                Jaccard_int[i].append(Jaccard_intersection[i])
                if i == 1:
                    if Jaccard[1][-1] > 0.8:
                        if 'sentPredFeatures' in review: 
                            output.write(str(review['sentences'])+'\n--\n'+str(review[predType])+'\n--\n'+str(review['sentPredFeatures'])+'\n====================\n\n')
            
            pre = 0.0
            rec = 0.0
            f1 = 0.0
            
            if (TP[i] + FN[i]):
                if (TP[i] + FP[i]):
                    pre = float(TP[i]) / (TP[i] + FP[i])
                else:
                    pre = 0.0
                rec = float(TP[i]) / (TP[i] + FN[i])
                if pre + rec:
                    f1 = 2 * pre * rec / (pre + rec)
                else:
                    f1 = 0.0
                
                Presision[i].append(pre)
                Recall[i].append(rec)
                F1[i].append(f1)
        
        
        
        '''
        Jaccard_vector
        '''
        for thres in np.arange(-0.05,1.05,0.05):
            Jaccard_vector[thres] = Jaccard_vector.get(thres, {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]})
            Accuracy_vector[thres] = Accuracy_vector.get(thres, {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]})
            
            for i in range(1,7):
                if Jaccard_vector_review[thres][i][1]:
                    Jaccard_vector[thres][i].append(Jaccard_vector_review[thres][i][0]/Jaccard_vector_review[thres][i][1])
                if Accuracy_vector_review[thres][i][1]:
                    Accuracy_vector[thres][i].append(Accuracy_vector_review[thres][i][0]/Accuracy_vector_review[thres][i][1])
        
        #print(aspectNum) 
        for r in aspectNum:
            aspectNumAvg[r].append(aspectNum[r])    
        
#        for i in range(1,4):
#            print(i, Jaccard_vector_review[0.5][i][0],Jaccard_vector_review[0.5][i][1],Jaccard_vector_review[0.5][i][0]/Jaccard_vector_review[0.5][i][1],Jaccard_intersection[i],Jaccard_union[i],Jaccard_intersection[i]/Jaccard_union[i])
#            print(len(Jaccard_vector[0.5][i]), len(Jaccard[i]), np.average(Jaccard_vector[0.5][i]), np.average(Jaccard[i]))
        
#        for r in RMSE_review:
#            if len(RMSE_review[r]):
#                RMSE_o[i].append(np.average(RMSE_review[r]))
        
        
        
#    print(TP_o)
    for i in range(1,7):
        Presision[i] = np.average(Presision[i])
        Recall[i] = np.average(Recall[i])
        F1[i] = np.average(F1[i])
        
        if (TP_o[i] + FP_o[i]):
            Presision_o[i] = float(TP_o[i]) / (TP_o[i] + FP_o[i])
        if (TP_o[i] + FN_o[i]):
            Recall_o[i] = float(TP_o[i])/ (TP_o[i] + FN_o[i])
        if Presision_o[i]+Recall_o[i]:
            F1_o[i] = 2 * Presision_o[i]* Recall_o[i] / (Presision_o[i]+Recall_o[i])
        
#        RMSE_o[i] = np.average(RMSE_o[i])
    
    PreRec = json.dumps([Presision, Recall, F1, Presision_o, Recall_o, F1_o])
    drawPR(y_true,y_pred_list, predType+' %d'%classes[1], path, classes)
    
    drawROC(y_true,y_pred_list, predType+' %d'%classes[1], path)
    
    
    
    J_v = {'th':[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    A_v = {'th':[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    
    for thres in np.arange(-0.05,1.05,0.05):
        J_v['th'].append(thres)
        A_v['th'].append(thres)
        for i in range(1,7):
            J_v[i].append(np.average(Jaccard_vector[thres][i]))
            A_v[i].append(np.average(Accuracy_vector[thres][i]))
#            if thres == 0.5:
#                print(i,thres,np.average(Jaccard_vector[thres][i]))
#                print(i,np.average(Jaccard[i]))
    
    drawJacAcc(J_v, A_v, predType+' %d'%classes[1], path)
    
    drawJaccDist(Jaccard, Jaccard_int, predType+' %d'%classes[1], path)
    
    for r in aspectNumAvg:
        aspectNumAvg[r] = np.average(aspectNumAvg[r])
    
    for i1 in range(1,7):
        for i2 in range(i1+1,7):
            print(i1,i2,stats.ttest_ind(Jaccard[i1],Jaccard[i2]))
    J = [np.average(Jaccard[i]) for i in range(1,7)]
    J_int = [np.average(Jaccard_int[i]) for i in range(1,7)]
    
#    RMSE_final = [np.average(RMSE[i]) for i in range(1,7)]
#    RMSE_o_final = [np.average(RMSE_o[i]) for i in range(1,7)]
#    RMSE = [RMSE_final,RMSE_o_final]
#    
#    print(RMSE)
    
    return J, J_int, PreRec, aspectNumAvg#, RMSE


def computeStat(path,modelfile, limit = np.Inf):
    logger = logging.getLogger('signature.computeStat')
    logger.info('start Stat computing')
    #get data
    r_file = path+'/specific_reviews_test_predictions.json'
    
    testReviews = list()
    for counter, line in enumerate(open(r_file,'r')):
        if not counter%1000:
            logger.debug('%d reviews loaded'%counter)
        if counter > limit:
            break
        testReviews.append(json.loads(line.strip()))
    logger.info('Reviews loaded')
    
    #load model
    modelDict = pickle.load(open(modelfile,'rb'))
    logger.info('Model loaded from %s'%modelfile)
    
    try:
        os.stat(path+'results/')
    except:
        os.mkdir(path+'results/')
    
    
    '''
    MF prediction
    '''
    MFpred = json.load(open(path+'reviews_test_exMFpred.json','r'))
    
    for r, review in enumerate(testReviews):
        reviewID = review['review_id']
        if 'exPredFeatures' in review:
            for aspect in review['exPredFeatures']:
                review['exPredFeatures'][aspect].append(MFpred[reviewID][aspect])
    
    
    #run function
    Jaccard1,J1_int, PreRec1, aspectNumAvg1 = computeStatWorker(testReviews, 'exPredFeatures', path, modelDict, classes = [0, 1])
    Jaccard0,J0_int, PreRec0, aspectNumAvg0 = computeStatWorker(testReviews, 'exPredFeatures', path, modelDict, classes = [1, 0])
#    print(PreRec1)
#    print(Jaccard1)
#    print(PreRec1)
    
    pred_index = range(1,7)
    #save Stat
    outfile = open(path+'/results/Jaccard_existence.txt','w')
    outfile.write('1:LogReg\t2:Random\t3:BusAvg\t4:Positive\t5:Negative\t6:MF\n')
    outfile.write('=====================\nClass 1\n=====================\n')
    outfile.write('Average number of good aspects in a review: %.2f; mentioned: %.2f\n%s\n\n'%(aspectNumAvg1['good'],aspectNumAvg1[0],
                                                                                              '\t'.join([str(k)+': %f'%float(aspectNumAvg1[k]) for k in pred_index])))
    outfile.write('Jaccard: \n%s\n'%'\t'.join([str(k+1)+': %f'%float(Jaccard1[k]) for k in range(len(Jaccard1))]))
    outfile.write('Jaccard average intersection: \n%s\n'%'\t'.join([str(k+1)+': %f'%float(J1_int[k]) for k in range(len(J1_int))]))
    res = json.loads(PreRec1)
    outfile.write('\nPrecision: \n%s\n'%'\t'.join([str(k)+': %f'%float(res[0][str(k)]) for k in pred_index]))
    outfile.write('\nRecall: \n%s\n'%'\t'.join([str(k)+': %f'%float(res[1][str(k)]) for k in pred_index]))
    outfile.write('\nF1: \n%s\n'%'\t'.join([str(k)+': %f'%float(res[2][str(k)]) for k in pred_index]))
    outfile.write('\nPrecision_o: \n%s\n'%'\t'.join([str(k)+': %f'%float(res[3][str(k)]) for k in pred_index]))
    outfile.write('\nRecall_o: \n%s\n'%'\t'.join([str(k)+': %f'%float(res[4][str(k)]) for k in pred_index]))
    outfile.write('\nF1_o: \n%s\n'%'\t'.join([str(k)+': %f'%float(res[5][str(k)]) for k in pred_index]))
#    outfile.write('\nRMSE: \n%s\n'%'\t'.join([str(k)+': %f'%float(RMSE[0][k]) for k in pred_index]))
#    outfile.write('\nRMSE by review: \n%s\n'%'\t'.join([str(k)+': %f'%float(RMSE[1][k]) for k in pred_index]))
    
    outfile.write('\n=====================\nClass 0\n=====================\n')
    outfile.write('Average number of good aspects in a review: %.2f; not mentioned: %.2f\n%s\n\n'%(aspectNumAvg0['good'],aspectNumAvg0[0],
                                                                                              '\t'.join([str(k)+': %f'%float(aspectNumAvg0[k]) for k in pred_index])))
    outfile.write('Jaccard: \n%s\n'%'\t'.join([str(k+1)+': %f'%float(Jaccard0[k]) for k in range(len(Jaccard0))]))
    outfile.write('Jaccard average intersection: \n%s\n'%'\t'.join([str(k+1)+': %f'%float(J0_int[k]) for k in range(len(J0_int))]))
    res = json.loads(PreRec0)
    outfile.write('\nPrecision: \n%s\n'%'\t'.join([str(k)+': %f'%float(res[0][str(k)]) for k in pred_index]))
    outfile.write('\nRecall: \n%s\n'%'\t'.join([str(k)+': %f'%float(res[1][str(k)]) for k in pred_index]))
    outfile.write('\nF1: \n%s\n'%'\t'.join([str(k)+': %f'%float(res[2][str(k)]) for k in pred_index]))
    outfile.write('\nPrecision_o: \n%s\n'%'\t'.join([str(k)+': %f'%float(res[3][str(k)]) for k in pred_index]))
    outfile.write('\nRecall_o: \n%s\n'%'\t'.join([str(k)+': %f'%float(res[4][str(k)]) for k in pred_index]))
    outfile.write('\nF1_o: \n%s\n'%'\t'.join([str(k)+': %f'%float(res[5][str(k)]) for k in pred_index]))
    
    outfile.close()
    
    
    
    '''
    MF prediction
    '''
    sentMFpred = json.load(open(path+'reviews_test_MFpred.json','r'))
    
    for r, review in enumerate(testReviews):
        reviewID = review['review_id']
        if 'sentPredFeatures' in review:
            for aspect in review['sentPredFeatures']:
                review['sentPredFeatures'][aspect].append(sentMFpred[reviewID][aspect])
    
    
    
    JaccardS1,JS1_int, PreRecS1, aspectNumAvgS1 = computeStatWorker(testReviews, 'sentPredFeatures', path, modelDict, classes = [0, 1])
    JaccardS0,JS0_int, PreRecS0, aspectNumAvgS0 = computeStatWorker(testReviews, 'sentPredFeatures', path, modelDict, classes = [1, 0])
    
#    print(JaccardS)
#    print(PreRecS)
    pred_index = range(1,7)
    #save Stat
    outfile = open(path+'/results/Jaccard_sentiment.txt','w')
    outfile.write('1:LogReg\t2:Random\t3:BusAvg\t4:Positive\t5:Negative\t6:MF\n')
    outfile.write('=====================\nClass 1\n=====================\n')
    outfile.write('Average number of aspects with sentiment in a review: %.2f; positive: %.2f\n%s\n\n'%(aspectNumAvgS1['good'],aspectNumAvgS1[0],
                                                                                              '\t'.join([str(k)+': %f'%float(aspectNumAvgS1[k]) for k in pred_index])))
    outfile.write('Jaccard: \n%s\n'%'\t'.join([str(k+1)+': %f'%float(JaccardS1[k]) for k in range(len(JaccardS1))]))
    outfile.write('Jaccard average intersection: \n%s\n'%'\t'.join([str(k+1)+': %f'%float(JS1_int[k]) for k in range(len(JS1_int))]))
    resS1 = json.loads(PreRecS1)
    outfile.write('\nPrecision: \n%s\n'%'\t'.join([str(k)+': %f'%float(resS1[0][str(k)]) for k in pred_index]))
    outfile.write('\nRecall: \n%s\n'%'\t'.join([str(k)+': %f'%float(resS1[1][str(k)]) for k in pred_index]))
    outfile.write('\nF1: \n%s\n'%'\t'.join([str(k)+': %f'%float(resS1[2][str(k)]) for k in pred_index]))
    outfile.write('\nPrecision_o: \n%s\n'%'\t'.join([str(k)+': %f'%float(resS1[3][str(k)]) for k in pred_index]))
    outfile.write('\nRecall_o: \n%s\n'%'\t'.join([str(k)+': %f'%float(resS1[4][str(k)]) for k in pred_index]))
    outfile.write('\nF1_o: \n%s\n'%'\t'.join([str(k)+': %f'%float(resS1[5][str(k)]) for k in pred_index]))
    
    outfile.write('\n=====================\nClass 0\n=====================\n')
    outfile.write('Average number of aspects with sentiment in a review: %.2f; negative: %.2f\n%s\n\n'%(aspectNumAvgS0['good'],aspectNumAvgS0[0],
                                                                                              '\t'.join([str(k)+': %f'%float(aspectNumAvgS0[k]) for k in pred_index])))
    outfile.write('Jaccard: \n%s\n'%'\t'.join([str(k+1)+': %f'%float(JaccardS0[k]) for k in range(len(JaccardS0))]))
    outfile.write('Jaccard average intersection: \n%s\n'%'\t'.join([str(k+1)+': %f'%float(JS0_int[k]) for k in range(len(JS0_int))]))
    resS0 = json.loads(PreRecS0)
    outfile.write('\nPrecision: \n%s\n'%'\t'.join([str(k)+': %f'%float(resS0[0][str(k)]) for k in pred_index]))
    outfile.write('\nRecall: \n%s\n'%'\t'.join([str(k)+': %f'%float(resS0[1][str(k)]) for k in pred_index]))
    outfile.write('\nF1: \n%s\n'%'\t'.join([str(k)+': %f'%float(resS0[2][str(k)]) for k in pred_index]))
    outfile.write('\nPrecision_o: \n%s\n'%'\t'.join([str(k)+': %f'%float(resS0[3][str(k)]) for k in pred_index]))
    outfile.write('\nRecall_o: \n%s\n'%'\t'.join([str(k)+': %f'%float(resS0[4][str(k)]) for k in pred_index]))
    outfile.write('\nF1_o: \n%s\n'%'\t'.join([str(k)+': %f'%float(resS0[5][str(k)]) for k in pred_index]))
    outfile.close()