# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 22:19:14 2022

@author: Gowri
"""

#Training dataset
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt
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
        data=pd.DataFrame(datamat['meas'])
    
        target=datamat['species']
        target_col=[]
        for i in target:
            target_col.append(i[0][0])

        data['target']=target_col

        headings ={0:"sepal length (cm)",1:"sepal width (cm)",2:"petal length (cm)",3:"petal width (cm)",4:"target"}
        data.rename(columns=headings,
          inplace=True)

         
    Train,Test = train_test_split(data,test_size=test_ratio,train_size=train_ratio,shuffle=True)
    Test, Validation = train_test_split(Test,test_size=test_ratio/2,train_size=test_ratio/2,shuffle=True)
    return Test,Train,Validation
def BT19ECE033_SVC():
    Test,Train,Validation = BT19ECE033_dataset_div_shuffle("fisheriris_matlab.mat",0.4,0.6)
    ###############Linear kernel###################
    print("LINEAR KERNEL")
    kernel = ['linear']
    C = [0.001,0.01,1,10,100]
    gamma = ['scale']
    # define grid search
    grid = dict(kernel=kernel,C=C,gamma=gamma)
    cv = 2
    grid_search = GridSearchCV(estimator=SVC(), param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(Validation[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]], Validation[["target"]])
#     print("Grid result:\n")
#     print(grid_result)
    # summarize results
    print("Best: \n Accuracy %f using prameters: %s" % (grid_result.best_score_, grid_result.best_params_))
#     print(grid_result.cv_results_.keys())
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    
#     poly_grid = {'accuracy':[],'params':[]}
    linear_grid = {'accuracy':[],'params':[]}
#     rbf_grid = {'accuracy':[],'params':[]}
      
    
    for mean, stdev, param in zip(means, stds, params):
        print("%f with: %r" % (mean, param))
#         if param['kernel']=='poly':
#             poly_grid['accuracy'].append(mean)
#             poly_grid['params'].append(param['C'])
        if param['kernel']=='linear':
            linear_grid['accuracy'].append(mean)
            linear_grid['params'].append(param['C'])
#         elif param['kernel']=='rbf':
#             rbf_grid['accuracy'].append(mean)
#             rbf_grid['params'].append(param['C'])

    test_grid = dict(kernel = [grid_result.best_params_['kernel']], C = [grid_result.best_params_['C']], gamma = [grid_result.best_params_['gamma']])
    test_grid_search = GridSearchCV(estimator=SVC(), param_grid=test_grid, n_jobs=-1, cv=2, scoring='accuracy',error_score=0)
    test_grid_result = test_grid_search.fit(Validation[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]], Validation[["target"]])
    pred_test = test_grid_result.predict(Test[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]])
    print("Accuracy on best parameters from linear kernel: ",test_grid_result.score(Test[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]], Test[["target"]]))
    print("Confusion matrix :\n",confusion_matrix(Test["target"],pred_test))
    
        ###############Polynomial kernel###################
    print("POLYNOMIAL KERNEL")
    kernel = ['poly']
    C = [0.001,0.01,1,10,100]
    gamma = ['scale']
    # define grid search
    grid = dict(kernel=kernel,C=C,gamma=gamma,degree=[3])
    cv = 2
    grid_search = GridSearchCV(estimator=SVC(), param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(Validation[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]], Validation[["target"]])
#     print("Grid result:\n")
#     print(grid_result)
    # summarize results
    print("Best: \n Accuracy %f using prameters: %s" % (grid_result.best_score_, grid_result.best_params_))
#     print(grid_result.cv_results_.keys())
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    
    poly_grid = {'accuracy':[],'params':[]}
#     linear_grid = {'accuracy':[],'params':[]}
#     rbf_grid = {'accuracy':[],'params':[]}
      
    
    for mean, stdev, param in zip(means, stds, params):
        print("%f with: %r" % (mean, param))
        if param['kernel']=='poly':
            poly_grid['accuracy'].append(mean)
            poly_grid['params'].append(param['C'])
#         elif param['kernel']=='linear':
#             linear_grid['accuracy'].append(mean)
#             linear_grid['params'].append(param['C'])
#         elif param['kernel']=='rbf':
#             rbf_grid['accuracy'].append(mean)
#             rbf_grid['params'].append(param['C'])

    test_grid = dict(kernel = [grid_result.best_params_['kernel']], C = [grid_result.best_params_['C']], gamma = [grid_result.best_params_['gamma']])
    test_grid_search = GridSearchCV(estimator=SVC(), param_grid=test_grid, n_jobs=-1, cv=2, scoring='accuracy',error_score=0)
    test_grid_result = test_grid_search.fit(Validation[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]], Validation[["target"]])
    pred_test = test_grid_result.predict(Test[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]])
    print("Accuracy on best parameters from polynomial kernel: ",test_grid_result.score(Test[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]], Test[["target"]]))
    print("Confusion matrix :\n",confusion_matrix(Test["target"],pred_test))
    
            ###############RBF kernel###################
    print("RBF KERNEL")
    kernel = ['rbf']
    C = [0.001,0.01,1,10,100]
    gamma = [0.1,1,10]
    # define grid search
    grid = dict(kernel=kernel,C=C,gamma=gamma)
    cv = 2
    grid_search = GridSearchCV(estimator=SVC(), param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(Validation[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]], Validation[["target"]])
#     print("Grid result:\n")
#     print(grid_result)
    # summarize results
    print("Best: \n Accuracy %f using prameters: %s" % (grid_result.best_score_, grid_result.best_params_))
#     print(grid_result.cv_results_.keys())
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    
#     poly_grid = {'accuracy':[],'params':[]}
#     linear_grid = {'accuracy':[],'params':[]}
    rbf_grid = {'accuracy':[],'params':[]}
      
    
    for mean, stdev, param in zip(means, stds, params):
        print("%f with: %r" % (mean, param))
#         if param['kernel']=='poly':
#             poly_grid['accuracy'].append(mean)
#             poly_grid['params'].append(param['C'])
#         elif param['kernel']=='linear':
#             linear_grid['accuracy'].append(mean)
#             linear_grid['params'].append(param['C'])
        if param['kernel']=='rbf':
            rbf_grid['accuracy'].append(mean)
            rbf_grid['params'].append(param['C'])

    test_grid = dict(kernel = [grid_result.best_params_['kernel']], C = [grid_result.best_params_['C']], gamma = [grid_result.best_params_['gamma']])
    test_grid_search = GridSearchCV(estimator=SVC(), param_grid=test_grid, n_jobs=-1, cv=2, scoring='accuracy',error_score=0)
    test_grid_result = test_grid_search.fit(Validation[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]], Validation[["target"]])
    pred_test = test_grid_result.predict(Test[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]])
    print("Accuracy on best parameters from RBF kernel: ",test_grid_result.score(Test[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]], Test[["target"]]))
    print("Confusion matrix :\n",confusion_matrix(Test["target"],pred_test))


        
    plt.figure()
    plt.plot(poly_grid['params'],poly_grid['accuracy'])
    plt.title("Accuracy vs hyperparameters for polynomial kernel")
    plt.figure()
    plt.plot(linear_grid['params'],linear_grid['accuracy'])
    plt.title("Accuracy vs hyperparameters for linear kernel")
    plt.figure()
    plt.plot(rbf_grid['params'],rbf_grid['accuracy'])
    plt.title("Accuracy vs hyperparameters for RBF kernel")
  
    
BT19ECE033_SVC()
