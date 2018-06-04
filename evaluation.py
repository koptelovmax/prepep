import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics

#from sklearn.ensemble import RandomForestClassifier
#from sklearn import svm
#from sklearn.naive_bayes import GaussianNB
#from sklearn import linear_model
#from sklearn.linear_model import SGDClassifier
#%%
def majority_vote_prediction(clf_list,test_X):
        
    test_results = np.zeros((np.size(test_X,-2),len(clf_list)),bool)
    for i in range(len(clf_list)):
        test_results[:,i] = clf_list[i].predict(test_X[i])
        
    return np.array([bool(round(np.mean(el))) for el in test_results])
#%%
# parameters:
num_molecules = 902
num_substructures = 100
num_samplings = 200
#total_num_features = 192

# load classes:
train_Y = np.load('classifier//training_data_Y.npy')

# load training data:
training_data_X = np.load('classifier//training_data_X.npy')
#training_data_num = np.load('training_data_num_'+str(j+1)+'.npy')

# initialize train and test sets:
## frequent substructures only:
train_X = np.zeros((num_molecules*2,num_substructures),bool)
  
## numerical features only:
#train_X = np.zeros((num_molecules*2,total_num_features),float)

## frequent substructures and numerical features:
#train_X = np.zeros((num_molecules*2,num_substructures+total_num_features),float)

# list of classifiers initialization:
clf_list = []

for i in range(num_samplings):
    print 'sampling',i+1
    
    # define train and test data: 
    ## frequent substructures only:
    train_X = training_data_X[i]
    
    ## numerical features only:
    #train_X = training_data_num[i]
    
    ## frequent substructures and numerical features:
    #train_X[:,0:num_substructures] = training_data_X[i]
    #train_X[:,num_substructures:num_substructures+total_num_features] = training_data_num[i]
    
    # define classifier:
    clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=17*3)    
    #clf = RandomForestClassifier(max_depth=4)
    #clf = svm.SVC(gamma=0.001,C=100)
    #clf = GaussianNB()
    #clf = linear_model.LogisticRegression()
    #clf = SGDClassifier()
    
    # perform learning:
    clf = clf.fit(train_X,train_Y)
         
    # save classifier:
    clf_list.append(clf)
#%%
# input file:
input_file = 'S9'#'S10','S11'

# dimensions of input data:
pred_X = np.load('evaluation//'+input_file+'_encoded.npy')

# number of molecules for classification:
num_molecules_classify = np.size(pred_X,1)

# results vector initialization:
pred_results = np.zeros(num_molecules_classify,bool)

# perform prediction:
pred_results = majority_vote_prediction(clf_list,pred_X)
#%%
# classes of input dataset:
pred_Y = np.zeros(num_molecules_classify,bool)
pred_Y[:] = True # S9, S11
#pred_Y[:] = False # S10

# verification:
false_positive,true_positive,thresholds = metrics.roc_curve(pred_Y,pred_results)
accuracy = 100-(np.sum(np.abs(pred_results-pred_Y))/float(num_molecules_classify))*100
auc_score = metrics.auc(false_positive,true_positive)
precision = metrics.precision_score(pred_Y,pred_results,average='binary')
recall = metrics.recall_score(pred_Y,pred_results,average='binary')

print 'accuracy',accuracy,'AUC',auc_score,'precision',precision,'recall',recall
#%%
# Confusion matrix:
TN,FP,FN,TP = metrics.confusion_matrix(pred_Y,pred_results).ravel()
print 'Confusion matrix:\n\n','FH-PAINS',TP,'FH-NoPAINS',FP,'\n','noneFH-PAINS',FN,'noneFH-NoPAINS',TN
#%%
# parse excel file for activity values of S9, S10
f = open('evaluation//activity_'+input_file+'.txt', 'r')

# activities vector initialization:
mol_acts = np.zeros(num_molecules_classify,np.float32)

count = 0
for line in f:
    try:
        mol_acts[count] = np.float32(line)
        count += 1
    except ValueError:
        print "Invalid input:", line

f.close()

# activity histograms
freq_act = []
infreq_act = []
for molecule in range(num_molecules_classify):
    if pred_results[molecule]:
        freq_act.append(mol_acts[molecule])
    else:
        infreq_act.append(mol_acts[molecule])

# plot activities histogram of FHs:
ax1 = plt.gca()
ax1.set_yscale('log')
ax1.set_ylim(0.75,5000)
plt.hist(freq_act,color='g',bins=50)
plt.title('Activity histogram of '+input_file+' dataset')
plt.ylabel('Frequency of freq-hitters (log)')
plt.xlabel('Percentage of activity')
#%%
# plot activities histogram of none-FHs:
ax1 = plt.gca()
ax1.set_yscale('log')
ax1.set_ylim(0.75,10000)
plt.hist(infreq_act,color='g',bins=50)
plt.title('Activity histogram of '+input_file+' dataset')
plt.ylabel('Frequency of infreq-hitters (log)')
plt.xlabel('Percentage of activity')
#%%