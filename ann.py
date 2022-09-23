# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 22:10:35 2022

@author: Gowri
"""

def BT19ECE033_dataset_div_shuffle(filepath,test_ratio,train_ratio):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import scipy.io
    
    data = pd.DataFrame()
    if(filepath[-3:]=="csv"):
        data = pd.read_csv(filepath)
    if(filepath[-4:]=="xlsx"):
        data = pd.read_excel(filepath)
    if(filepath[-3:]=="mat"):
        datamat = scipy.io.loadmat(filepath)
#         data = datamat
        data = pd.DataFrame(list(zip(datamat["t"][0],datamat["t"][1],datamat["x"].T)))
        conditions = [(data[0]==1) & (data[1]==0),
                       (data[0]==0) & (data[1]==1)]
        values = ["1","0"]
        
        data["Detected"] = np.select(conditions,values)
        patient_data = []
         
    Train,Test = train_test_split(data,test_size=test_ratio,train_size=train_ratio,shuffle=True)
    Test, Validation = train_test_split(Test,test_size=test_ratio/2,train_size=test_ratio/2,shuffle=True)
    return Train,Validation,Test


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def BT19ECE033_ANN():
    BT19ECE033_ANN_WithoutHiddenLayer()
    print("Artificial Neural Network with 1 hidden layer")
    Train,Validation,Test = BT19ECE033_dataset_div_shuffle("Matlab_cancer.mat",0.4,0.6)
    #216 patients
    # X = 1x100
    #1 hidden layer 
    print("Train data shape,Validation data shape,Test data shape : ",Train.shape,Validation.shape,Test.shape)
    
    W_ih = 2*np.random.random((len(Train[2][0]),(len(Train[2]))))-1
    W_ho = 2*np.random.random(((len(Train[2])),1))-1
    
#    print("Input weights: ",W_ih.shape)
#    print("output weights: ",W_ho.shape)
      
        
    W_ih,W_ho = network_train(1,np.array(list(Train[2])),np.array(list(Train["Detected"])).astype(int),W_ih,W_ho)

    #running the trained network with modified weights
    validation_op = execute_ann(np.array(list(Validation[2])),W_ih,W_ho)
    train_op = execute_ann(np.array(list(Train[2])),W_ih,W_ho)
    test_op = execute_ann(np.array(list(Test[2])),W_ih,W_ho)
    
    actual_validation_op = np.array(list(Validation["Detected"]))
    actual_validation_op = actual_validation_op.reshape(actual_validation_op.shape[0],1)
    actual_validation_op = [int(a) for a in actual_validation_op]

    actual_train_op = np.array(list(Train["Detected"]))
    actual_train_op = actual_train_op.reshape(actual_train_op.shape[0],1)
    actual_train_op = [int(a) for a in actual_train_op]
    
    actual_test_op = np.array(list(Test["Detected"]))
    actual_test_op = actual_test_op.reshape(actual_test_op.shape[0],1)
    actual_test_op = [int(a) for a in actual_test_op]

    error_train = MSE(actual_train_op,train_op)
    error_validation = MSE(actual_validation_op,validation_op)
    error_test = MSE(actual_test_op,test_op)
    
    new_error_validation = error_validation
    i = 3
    
    
    #Avoiding overfitting using early stopping   
    while new_error_validation>=error_validation and i<=100:
        W_ih,W_ho = network_train(i,np.array(list(Train[2])),np.array(list(Train["Detected"])).astype(int),W_ih,W_ho)
        validation_op = execute_ann(np.array(list(Validation[2])),W_ih,W_ho)
        
        actual_validation_op = np.array(list(Validation["Detected"]))
        actual_validation_op = actual_validation_op.reshape(actual_validation_op.shape[0],1)
        actual_validation_op = [int(a) for a in actual_validation_op]
        
        new_error_validation = MSE(actual_validation_op,validation_op)
        i = i+1
    
    
    print("Train error, Validation error, Validation error after early stopping : ",error_train,error_validation,new_error_validation)
    
    #Testing data
    test_op = execute_ann(np.array(list(Test[2])),W_ih,W_ho)
    
    actual_test_op = np.array(list(Test["Detected"]))
    actual_test_op = actual_test_op.reshape(actual_test_op.shape[0],1)
    actual_test_op = [int(a) for a in actual_test_op]
    
    error_test = MSE(actual_test_op,test_op)
       
    print("Error after running network on test data : ",error_test)
    test_op[test_op<np.mean(test_op)]=0
    test_op[test_op>=np.mean(test_op)]=1
    print("Testing pred and actual op: ",test_op.reshape(len(test_op),),actual_test_op)
    tn, fp, fn, tp = confusion_matrix(actual_test_op, test_op).ravel()
    print("Confusion matrix values : tn tp fn fp : ",tn,tp,fn,fp)
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    print("Sensitivity: ",sensitivity)
    print("Specificity: ",specificity)
    fpr, tpr, _ = metrics.roc_curve(actual_test_op,test_op)
    auc = metrics.roc_auc_score(actual_test_op, test_op)

    #create ROC curve
    plt.figure()
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)

    plt.show()


def MSE(y, Y):
#     y : actual label (matrix)
#     Y : predicted label (matrix)
    
    return np.mean((y - Y)**2)


def execute_ann(X,W_ih,W_ho):
    H1 = 1/(1+np.exp(-(np.dot(X,W_ih))))
    op = 1/(1+np.exp(-(np.dot(H1,W_ho))))
    return op

def network_train(n,X,Y,W_ih,W_ho,shapes=True):
    if shapes==True:
        for j in range(n):
            H1 = 1/(1+np.exp(-(np.dot(X,W_ih))))
            output = 1/(1+np.exp(-(np.dot(H1,W_ho))))

            delta_o = (Y.reshape((Y.shape[0],1))-output)*(output*(1-output))

            delta_H1 = delta_o.T.dot(W_ho) * (H1 * (1-H1))
            W_ho = W_ho + H1.dot(delta_o)
            W_ih = W_ih + X.T.dot(delta_H1)
#             print("final weights shapes: ih and ho",W_ih.shape,W_ho.shape)
#     else:
#         for j in range(n):
#             H1 = 1/(1+np.exp(-(np.dot(X,W_ih.T))))
#             output = 1/(1+np.exp(-(np.dot(H1.T,W_ho))))
# #             print(output,Y)
#             delta_o = (Y-output)*(output*(1-output))
#             delta_H1 = delta_o.dot(W_ho) * (H1 * (1-H1))
#             W_ho = W_ho + H1.dot(delta_o)
#             W_ih = W_ih + X.T.dot(delta_H1)
    return W_ih,W_ho
        


def BT19ECE033_ANN_WithoutHiddenLayer():
    print("Artificial Neural Network without hidden layer")
    Train,Validation,Test = BT19ECE033_dataset_div_shuffle("Matlab_cancer.mat",0.4,0.6)
    #216 patients
    # X = 1x100
    #1 hidden layer 
    print("Train data shape,Validation data shape,Test data shape : ",Train.shape,Validation.shape,Test.shape)
    
    W = 2*np.random.random(((len(Train[2][0])),1))-1
    
#    print("Input weights: ",W.shape)    
        
    W = network_train_nolayer(1,np.array(list(Train[2])),np.array(list(Train["Detected"])).astype(int),W)

    #running the trained network with modified weights
    validation_op = execute_ann_nolayer(np.array(list(Validation[2])),W)
    train_op = execute_ann_nolayer(np.array(list(Train[2])),W)
    test_op = execute_ann_nolayer(np.array(list(Test[2])),W)
    
    actual_validation_op = np.array(list(Validation["Detected"]))
    actual_validation_op = actual_validation_op.reshape(actual_validation_op.shape[0],1)
    actual_validation_op = [int(a) for a in actual_validation_op]

    actual_train_op = np.array(list(Train["Detected"]))
    actual_train_op = actual_train_op.reshape(actual_train_op.shape[0],1)
    actual_train_op = [int(a) for a in actual_train_op]
    
    actual_test_op = np.array(list(Test["Detected"]))
    actual_test_op = actual_test_op.reshape(actual_test_op.shape[0],1)
    actual_test_op = [int(a) for a in actual_test_op]

    error_train = MSE(actual_train_op,train_op)
    error_validation = MSE(actual_validation_op,validation_op)
    error_test = MSE(actual_test_op,test_op)
    
    new_error_validation = error_validation
    i = 3
    
    
    #Avoiding overfitting using early stopping   
    while new_error_validation>=error_validation and i<=100:
        W = network_train_nolayer(i,np.array(list(Train[2])),np.array(list(Train["Detected"])).astype(int),W)
        validation_op = execute_ann_nolayer(np.array(list(Validation[2])),W)
        
        actual_validation_op = np.array(list(Validation["Detected"]))
        actual_validation_op = actual_validation_op.reshape(actual_validation_op.shape[0],1)
        actual_validation_op = [int(a) for a in actual_validation_op]
        
        new_error_validation = MSE(actual_validation_op,validation_op)
        i = i+1
    
    
    print("Train error, Validation error, Validation error after early stopping : ",error_train,error_validation,new_error_validation)
    
    #Testing data
    test_op = execute_ann_nolayer(np.array(list(Test[2])),W)
    
    actual_test_op = np.array(list(Test["Detected"]))
    actual_test_op = actual_test_op.reshape(actual_test_op.shape[0],1)
    actual_test_op = [int(a) for a in actual_test_op]
    
    error_test = MSE(actual_test_op,test_op)
       
    print("Testing error: ",error_test)
#     print("Testing pred and actual op: ",test_op,actual_test_op)
    test_op[test_op<np.mean(test_op)]=0
    test_op[test_op>=np.mean(test_op)]=1
    print("Testing pred and actual op: ",test_op.reshape(len(test_op),),actual_test_op)
    tn, fp, fn, tp = confusion_matrix(actual_test_op, test_op).ravel()
    print("tn tp fn fp",tn,tp,fn,fp)
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    print("Sensitivity: ",sensitivity)
    print("Specificity: ",specificity)
    fpr, tpr, _ = metrics.roc_curve(actual_test_op,test_op)
    auc = metrics.roc_auc_score(actual_test_op, test_op)

    #create ROC curve
    plt.figure()
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.title("Without hidden layer")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)

    plt.show()


#def MSE(y, Y):
##     y : actual label (matrix)
##     Y : predicted label (matrix)
#    
#    return np.mean((y - Y)**2)


def execute_ann_nolayer(X,W):
    op = 1/(1+np.exp(-(np.dot(X,W))))
#     op = 1/(1+np.exp(-(np.dot(H1,W_ho))))
    return op

def network_train_nolayer(n,X,Y,W,shapes=True):
    if shapes==True:
        for j in range(n):
#             print("X and Wih",X.shape,W_ih.shape)
            output = 1/(1+np.exp(-(np.dot(X,W))))
            delta_o = (Y.reshape((Y.shape[0],1))-output)*(output*(1-output))
            W = W + X.T.dot(delta_o)
    return W
        

BT19ECE033_ANN()
