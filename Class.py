# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 21:38:08 2019

@author: ahmed
"""
import numpy as np
from numpy.linalg import norm


class ConvoClassifier:

    def __init__(self):
        
        print('Object created')
        
    
    def __splitN(self,X,y):
    
        uniq_val = np.unique(y)
        container = []
        for i in range(uniq_val.shape[0]):
            #        X_temp = X[np.where(y==uniq_val[i])[0]]
            index = np.where(y==uniq_val[i])[0]
            container.append(index)
        
        return container
    
    def fit(self,X,y):
        container_index = self.__splitN(X,y)
        self.__Fneg = X[container_index[0]]
        self.__Fpos = X[container_index[1]]
        
        return
    
    def predict(self,X_test):
        if np.ndim(X_test) == 1:
            m = 1
            X_test = X_test[None,:]
        else:
            m = X_test.shape[0]
            
        convo_pos = []
        convo_neg = []
        dist_pos = []
        dist_neg = []
        for k in range(m):
            convo_pos_temp = 0
            convo_neg_temp = 0
            dist_pos_temp = 0
            dist_neg_temp = 0
            for i in range(self.__Fpos.shape[0]):
                convo_pos_temp += np.sum(np.convolve(X_test[k],self.__Fpos[i]))
                dist_pos_temp += norm(X_test[k]-self.__Fpos[i])
            for j in range(self.__Fneg.shape[0]):
                convo_neg_temp += np.sum(np.convolve(X_test[k],self.__Fneg[j]))
                dist_neg_temp += norm(X_test[k]-self.__Fneg[j])
                
            convo_pos.append(convo_pos_temp)
            convo_neg.append(convo_neg_temp)
            dist_pos.append(dist_pos_temp)
            dist_neg.append(dist_neg_temp)
    
        convo_pos = np.asarray(convo_pos)
        dist_pos = np.asarray(dist_pos)
        convo_neg = np.asarray(convo_neg)
        dist_neg = np.asarray(dist_neg)
        theta0 = convo_pos/dist_pos
        theta1 = convo_neg/dist_neg
        return (theta0 > theta1)*1
#        return theta1,theta2
#        return theta0,theta1
