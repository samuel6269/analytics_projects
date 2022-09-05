# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 23:22:03 2021

@author: anawaz
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report, accuracy_score

def plot_decision_boundary(model, title, x_train, x_test, y_train, y_test):
    cmap_light = ListedColormap(['orange', 'cornflowerblue'])
    cmap_bold = ['darkorange', 'darkblue']
    h = 0.01 # step size for mesh
    # cmap_light = ListedColormap(['#FFAAAA',  '#AAAAFF'])
    # cmap_bold = ListedColormap(['#FF0000',  '#0000FF'])

    x_min, x_max = x_train[:,0].min()-0.5, x_train[:,0].max()+0.5
    y_min, y_max = x_train[:,1].min()-0.5, x_train[:,1].max()+0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # plot the training points
    ax = sns.scatterplot(x=x_train[:, 0], y=x_train[:, 1], hue=y_train, palette=cmap_bold, alpha=0.20, edgecolor="black", legend='full')
    legend_labels, _ = ax.get_legend_handles_labels()
    ax.legend(legend_labels,['RTS','Pass'])
    sns.scatterplot(x=x_test[:,0], y=x_test[:,1], hue=y_test, palette=cmap_bold, alpha=1, edgecolor="purple", legend=False)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    filen = title + "_db.png"
    plt.savefig(filen, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_cv_lambda(filen, title, x_name, alpha_vals, cv_scores):
    # filen string should have ".png" at the end
    plt.figure()
    plt.plot(alpha_vals, cv_scores)
    plt.xlabel(x_name)
    plt.ylabel('Accuracy Rate')
    plt.title(title)
    plt.savefig(filen, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def data_Xy_split(fname):
    df = pd.read_csv(fname)
    data = np.array(df.iloc[:, 1:])
    data[:, 2] = np.where(data[:, 2] == 'A', 0, 1)

    X = data[:, 0:2]
    y = data[:, -1].astype('int')
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=3)
    return Xtrain, Xtest, ytrain, ytest

def plot_ROC_curve(cls, cls_name, X_test, y_test):
    plot_roc_curve(cls, X_test, y_test)
    plt.title('{} ROC Curve'.format(cls_name))
    plt.savefig('{}_roc.png'.format(cls_name), dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def report_accuracy_score(cls, cls_name, X_test, y_test):
    y_predict = cls.predict(X_test)
    #acc_score = accuracy_score(y_test,y_predict)
    results = classification_report(y_test, y_predict, output_dict = True)
    print('{} Accuracy Score: {}'.format(cls_name,results['accuracy']))
    #print('{} Class 0 Precision: {}'.format(cls_name,results['0']['precision']))
    print('{} Class 1 Precision: {}'.format(cls_name, results['1']['precision']))
    print('\n')

# data = pd.read_csv('rts.csv').to_numpy()
# data[data[:,3]=='A',3]='RTS'
# data[data[:,3]=='N',3]='Good'
#
# Xtrain, Xtest, ytrain, ytest = data_Xy_split('rts.csv')
#
# sns.scatterplot(x=data[:,1], y=data[:,2], hue=data[:,3])
# plt.xlabel("V1")
# plt.ylabel("V2")
# plt.savefig('rts_plot', dpi=150, bbox_inches='tight', pad_inches=0.1)
# plt.close()
#
# plt.figure(1)
# sns.scatterplot(x=Xtrain[:,0], y=Xtrain[:,1], hue=ytrain)
# plt.xlabel("V1")
# plt.ylabel("V2")
# plt.xlim(-1,1.25)
# plt.ylim(-1, 1.75)
# plt.savefig('rts_plot_train', dpi=150, bbox_inches='tight', pad_inches=0.1)
# plt.close(1)

# plt.figure(2)
# sns.scatterplot(x=Xtest[:,0], y=Xtest[:,1], hue=ytest)
# plt.xlabel("V1")
# plt.ylabel("V2")
# plt.xlim(-1,1.25)
# plt.ylim(-1, 1.75)
# plt.savefig('rts_plot_test', dpi=150, bbox_inches='tight', pad_inches=0.1)
# plt.close(2)

# plt.figure(3)
# cmap_light = ['darkorange', 'cornflowerblue']
# ax = sns.scatterplot(x=Xtrain[:,0], y=Xtrain[:,1], hue=ytrain, alpha=0.6, palette=cmap_light, edgecolor="black")
# legend_labels, _ = ax.get_legend_handles_labels()
# ax.legend(legend_labels,['RTS','Pass'])
# sns.scatterplot(x=Xtest[:,0], y=Xtest[:,1], hue=ytest, legend=False, alpha=0.6, palette=cmap_light, edgecolor="black")
# # sns.scatterplot(x=Xtest[:,0], y=Xtest[:,1], hue=ytest, legend=False, alpha=0.5, edgecolor="purple")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title('Dataset')
# x_min, x_max = Xtrain[:,0].min()-0.5, Xtrain[:,0].max()+0.5
# y_min, y_max = Xtrain[:,1].min()-0.5, Xtrain[:,1].max()+0.5
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# # plt.xlim(-1,1.25)
# # plt.ylim(-1, 1.75)
# plt.savefig('rts_data.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
# plt.close(3)
