import numpy as np
from sklearn import tree
from sklearn import metrics

#from sklearn.ensemble import RandomForestClassifier
#from sklearn import svm
#from sklearn.naive_bayes import GaussianNB
#from sklearn import linear_model
#from sklearn.linear_model import SGDClassifier
#%%
# parameters:
num_molecules = 812
num_substructures = 100
num_folds = 10
num_samplings = 200
num_molecules_test = 90
num_infreq_molecules_test = 90 #90/900/9000
total_num_features = 192

# initialization:
accuracy = np.zeros(num_folds,float)
auc_score = np.zeros(num_folds,float)
precision = np.zeros(num_folds,float)
recall = np.zeros(num_folds,float)

# load classes:
train_Y = np.load('training_data_Y.npy')
test_data_Y = np.load('test_data_Y.npy')
#%%
def majority_vote_prediction(clf_list,test_X):
        
    test_results = np.zeros((np.size(test_X,-2),len(clf_list)),bool)
    for i in range(len(clf_list)):
        test_results[:,i] = clf_list[i].predict(test_X[i])
               
    return [bool(round(np.mean(el))) for el in test_results]
#%%
def cstd(x): # corrected sample standard deviation
    
    return np.sqrt(np.sum((x-np.mean(x))**2)/float(len(x)-1))
#%%
test_results = np.zeros((num_folds,num_molecules_test+num_infreq_molecules_test),bool)

# classes of test data:
test_Y = np.zeros(num_molecules_test+num_infreq_molecules_test,bool)
test_Y = test_data_Y

training_errors = []

for j in range(10):
    print 'fold',j+1
    
    # load training and test data:
    training_data_X = np.load('training_data_X_'+str(j+1)+'.npy')
    test_data_X = np.load('test_data_X_'+str(j+1)+'.npy')
    training_data_num = np.load('training_data_num_'+str(j+1)+'.npy')
    test_data_num = np.load('test_data_num_'+str(j+1)+'.npy')
    
    # initialize train and test sets (uncomment what necessary):
    ## frequent substructures only:
    train_X = np.zeros((num_molecules*2,num_substructures),bool)
    test_X = np.zeros((num_samplings,num_molecules_test+num_infreq_molecules_test,num_substructures),bool)
        
    ## numerical features only:
    #train_X = np.zeros((num_molecules*2,total_num_features),float)
    #test_X = np.zeros((num_samplings,num_molecules_test+num_infreq_molecules_test,total_num_features),float)

    ## frequent substructures and numerical features:
    #train_X = np.zeros((num_molecules*2,num_substructures+total_num_features),float)
    #test_X = np.zeros((num_samplings,num_molecules_test+num_infreq_molecules_test,num_substructures+total_num_features),float)

    # list of classifiers initialization:
    clf_list = []
    
    # training errors inside one fold:    
    fold_errors = []

    for i in range(num_samplings):
        #print 'experiment',i+1
            
        # define train and test data (uncomment what necessary): 
        ## frequent substructures only:
        train_X = training_data_X[i]
        test_X[i] = test_data_X[i]
            
        ## numerical features only:
        #train_X = training_data_num[i]
        #test_X[i] = test_data_num
            
        ## frequent substructures and numerical features:
        #train_X[:,0:num_substructures] = training_data_X[i]
        #train_X[:,num_substructures:num_substructures+total_num_features] = training_data_num[i]
        #test_X[i,:,0:num_substructures] = test_data_X[i]
        #test_X[i,:,num_substructures:num_substructures+total_num_features] = test_data_num
        
        # define classifier (uncomment what necessary):
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
        
        # collect training error:
        fold_errors.append(clf.score(train_X,train_Y))

    # save training errors for every fold:    
    training_errors.append(fold_errors)
    
    # prediction:
    test_results[j] = majority_vote_prediction(clf_list,test_X)
       
    # verification:
    false_positive,true_positive,thresholds = metrics.roc_curve(test_Y,test_results[j])
    accuracy[j] = 100-(np.sum(np.abs(test_results[j]-test_Y))/float(num_molecules_test+num_infreq_molecules_test))*100
    auc_score[j] = metrics.auc(false_positive,true_positive)
    precision[j] = metrics.precision_score(test_Y,test_results[j], average='binary')
    recall[j] = metrics.recall_score(test_Y,test_results[j],average='binary')

#np.save('test_results.npy',test_results)
#np.save('auc_score.npy',auc_score)
#np.save('precision.npy',precision)
#np.save('recall.npy',recall)

print 'Average accuracy:',np.mean(accuracy),'cstd:',cstd(accuracy)
print 'Average AUC score:',np.mean(auc_score),'cstd:',cstd(auc_score)
print 'Average precision:',np.mean(precision),'cstd:',cstd(precision)
print 'Average recall:',np.mean(recall),'cstd:',cstd(recall)
print 'Line to copy for results.xls:'
print round(np.mean(accuracy),2),'\t',round(cstd(accuracy),2),'\t',round(np.mean(auc_score),4),'\t',round(cstd(auc_score),4),'\t',round(np.mean(precision),4),'\t',round(cstd(precision),4),'\t',round(np.mean(recall),4),'\t',round(cstd(recall),4)
#%%