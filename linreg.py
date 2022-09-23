# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 23:33:59 2022

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
        data = datamat["accidents"]
        heading = data[0][0][3][0]
        head = []
        for ele in range(0,len(heading)):
            head.append(heading[ele][0]) 
        pd.set_option('display.max_rows',10 )

        data = pd.DataFrame(data[0][0][2], columns=head)
        data = data.iloc[:,[3,13]]
         
    Train,Test = train_test_split(data,test_size=test_ratio,train_size=train_ratio,shuffle=True)
    return Train,Test
#
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#Train,Test = BT19ECE033_dataset_div_shuffle("Matlab_accidents.mat",0.3,0.7)
#Y = Train.to_numpy()[:,0]
#Y = Y.reshape(len(Y),1)
#X = Train.to_numpy()[:,1:]
#X_Train = np.hstack([np.ones((len(X),1)),np.array(X).reshape(-1,1)])
#X_Test = Test.to_numpy()[:,1:]
## X_Test = np.hstack([np.ones((len(X_Test),1)),np.array(X_Test).reshape(-1,1)])
#Y_Test = Test.to_numpy()[:,0]
#Y_Test = Y_Test.reshape(len(Y_Test),1)

#pseudo inverse
def pseudo_inverse(X,X_Train,Y,X_Test,Y_Test):
    W = np.linalg.pinv(X_Train.T @ X_Train) @ X_Train.T @ Y
    Y_hat = X_Train@W
    plt.plot(X,Y,"ro")
    plt.plot(X_Test,Y_Test,"ro")
    plt.title("Complete dataset scatter plot")
    plt.figure()
    plt.plot(X,Y_hat)
    plt.plot(X,Y,"*")
    plt.legend(["Pseudo inverse plot","Data"])
    plt.title("Pseudo inverse")
    plt.figure()
    return W

def grad_descent(X,X_Train,Y,W,L = 0.0001, i = 100):
    weight = np.random.rand(X_Train.shape[1],1)
    max_x = np.max(X_Train[:,1])
    X_Train_norm = X_Train
    X_Train_norm[:,1]= X_Train_norm[:,1]/max_x
    max_y = np.max(Y)
    Y_norm = Y/max_y
#     plt.plot(max_x*normalize_train_x[:,1], max_y*normalize_train_y, 'ro')
    # print(normalize_train_x, normalize_train_y)
    for ele in range(i):
        y_hat = X_Train_norm @ weight
        diff = y_hat - Y_norm.reshape(-1,1)
        loss = np.sum(diff**2)/len(Y_norm)
        dl_da = 2/len(Y_norm)
        da_ddiff  = diff
        ddiff_dy = 1
        ddiff_dw= X_Train_norm.T
        dldw = dl_da * ddiff_dy*(ddiff_dw @ da_ddiff)
        weight = weight - L*dldw
#     plt.plot(X,(X_Train@W))
    plt.plot(max_x*X_Train_norm[:,1], max_y*Y_norm, 'ro')
    plt.plot(max_x*X_Train_norm[:,1], max_y*(X_Train_norm @ weight))
    plt.legend(["Data", "Linear Regression"])
    plt.title("Gradient descent optimisation")
    plt.figure()
#     plt.plot(loss_over_time)
#     plt.legend(["Loss"])
    # plt.show()
    return weight
    
def BT19ECE033_linreg(path,train_r,test_r):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    Train,Test = BT19ECE033_dataset_div_shuffle(path,train_r,test_r)
    Y = Train.to_numpy()[:,0]
    Y = Y.reshape(len(Y),1)
    X = Train.to_numpy()[:,1:]
    X_Train = np.hstack([np.ones((len(X),1)),np.array(X).reshape(-1,1)])
    X_Test = Test.to_numpy()[:,1:]
    # X_Test = np.hstack([np.ones((len(X_Test),1)),np.array(X_Test).reshape(-1,1)])
    Y_Test = Test.to_numpy()[:,0]
    Y_Test = Y_Test.reshape(len(Y_Test),1)
    W = pseudo_inverse(X,X_Train,Y,X_Test,Y_Test)
    weight = grad_descent(X,X_Train,Y,W)
    
BT19ECE033_linreg("Matlab_accidents.mat",0.3,0.7)

