# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:35:54 2021

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV,LeaveOneOut
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

import rts as util

def banner(clf_name):
    print('')
    print('*'*len(clf_name))
    print(clf_name)
    print('*'*len(clf_name))

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = util.data_Xy_split('rts.csv')
    
    # SVM-linear
    banner('SVM Linear Results')

    clf_svm = SVC()
    
    loo = LeaveOneOut()
    parameters = {'kernel':['linear'], 'C':np.arange(0.1,3,0.2)}
    clf = GridSearchCV(clf_svm, parameters,cv=loo)
    clf.fit(X_train, y_train)
    results = np.array(pd.DataFrame(clf.cv_results_)[['param_C','mean_test_score','std_test_score']])
    print(results)
    C_param = results[np.argmax(results[:,1]),0] #find C with best CV accuracy score
    clf_svm_l = SVC(kernel='linear', C=C_param).fit(X_train, y_train)
    util.plot_cv_lambda('SVM_L_CV.png','SVM Linear CV Results', 'C', results[:,0], results[:,1])
    
    util.plot_decision_boundary(clf_svm_l, 'SVM Linear', X_train, X_test, y_train, y_test)
    util.plot_ROC_curve(clf_svm_l,'SVM Linear Classifier', X_test, y_test)
    util.report_accuracy_score(clf_svm_l, 'SVM Linear Classifier', X_test, y_test)

    # SVM-rbf    
    banner('SVM RBF Results')
    
    parameters = {'kernel':['rbf'], 'gamma':[0.0001,0.001,0.005,0.01,0.04,0.05,0.06,0.07,0.1,0.5,1,2,3,4,5,7,10]}
    clf = GridSearchCV(clf_svm, parameters,cv=loo)
    clf.fit(X_train, y_train)
    results = np.array(pd.DataFrame(clf.cv_results_)[['param_gamma','mean_test_score','std_test_score']])
    print(results)
    gamma_param = results[np.argmax(results[:,1]),0] #find gamma with best CV accuracy score
    clf_svm_r = SVC(kernel='rbf', gamma=gamma_param).fit(X_train, y_train)
    util.plot_cv_lambda('SVM_R_CV.png','SVM RBF CV Results', 'Gamma', results[:,0], results[:,1])
    
    util.plot_decision_boundary(clf_svm_r, 'SVM RBF', X_train, X_test, y_train, y_test)
    util.plot_ROC_curve(clf_svm_r,'SVM RBF Classifier', X_test, y_test)
    util.report_accuracy_score(clf_svm_r, 'SVM RBF Classifier', X_test, y_test)
    
    
    # CART-Gini
    banner('CART Gini Results')
    
    clf_cart_g = DecisionTreeClassifier()
    parameters = {'min_samples_leaf':np.arange(1,11)}
    clf = GridSearchCV(clf_cart_g, parameters,cv=loo)
    clf.fit(X_train, y_train)
    results = np.array(pd.DataFrame(clf.cv_results_)[['param_min_samples_leaf','mean_test_score','std_test_score']])
    print(results)
    min_samples_param = results[np.argmax(results[:,1]),0] #find leaf size with best CV accuracy score
    clf_cart_g = DecisionTreeClassifier(min_samples_leaf=min_samples_param).fit(X_train, y_train)
    plt.figure(figsize=(20, 25), dpi=150)
    plot_tree(clf_cart_g, filled=True)
    plt.savefig('cart_plot_tree.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    util.plot_cv_lambda('CART_CV.png','CART CV Results', 'Min Leaf Size', results[:,0], results[:,1])
    
    # print('')
    # print('CART Accuracy {:.3f}'.format(clf_cart_g.score(X, y)))
    
    util.plot_decision_boundary(clf_cart_g, 'CART', X_train, X_test, y_train, y_test)
    util.plot_ROC_curve(clf_cart_g,'CART Classifier', X_test, y_test)
    util.report_accuracy_score(clf_cart_g, 'CART Classifier', X_test, y_test)
    
    # CART-Entropy
    # banner('CART Entropy Results')
    
    # clf_cart_e = DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)
    # parameters = {'criterion':['entropy'],'min_samples_leaf':np.arange(1,11)}
    # clf = GridSearchCV(clf_cart_e, parameters,cv=loo)
    # clf.fit(X_train, y_train)
    # results = np.array(pd.DataFrame(clf.cv_results_)[['param_min_samples_leaf','mean_test_score','std_test_score']])
    # print(results)
    # min_samples_param = results[np.argmax(results[:,1]),0] #find leaf size with best CV accuracy score
    # clf_cart_e = DecisionTreeClassifier(min_samples_leaf=min_samples_param, criterion='entropy').fit(X_train, y_train)
    # plt.figure(figsize=(20, 25), dpi=150)
    # plot_tree(clf_cart_e, filled=True)
    # plt.savefig('cart_e_plot_tree.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    # plt.close()
    # util.plot_cv_lambda('CART_e_CV.png','CART CV Results', 'Min Leaf Size', results[:,0], results[:,1])
    
    # # print('')
    # # print('CART Accuracy {:.3f}'.format(clf_cart_g.score(X, y)))
    
    # util.plot_decision_boundary(clf_cart_e, 'CART_e', X_train, X_test, y_train, y_test)
    # util.plot_ROC_curve(clf_cart_e,'CART_e Classifier', X_test, y_test)
    # util.report_accuracy_score(clf_cart_e, 'CART Classifier', X_test, y_test)    
    
    # Logistic Regression
    banner('Logistic Results')
    
    clf_logist = LogisticRegression()
    
    parameters = {'solver':['liblinear'], 'C':np.arange(0.1,5,0.5)}
    clf = GridSearchCV(clf_logist, parameters,cv=loo)
    clf.fit(X_train, y_train)
    results = np.array(pd.DataFrame(clf.cv_results_)[['param_C','mean_test_score','std_test_score']])
    print(results)
    C_param = results[np.argmax(results[:,1]),0] #find C with best CV accuracy score
    clf_logist = LogisticRegression(solver='liblinear', C=C_param).fit(X_train, y_train)
    util.plot_cv_lambda('Logistic_CV.png','Logistic Reg CV Results', 'C', results[:,0], results[:,1])
    
    util.plot_decision_boundary(clf_logist, 'Logistic', X_train, X_test, y_train, y_test)
    util.plot_ROC_curve(clf_logist,'Logistic Classifier', X_test, y_test)
    util.report_accuracy_score(clf_logist, 'Logistic Classifier', X_test, y_test)