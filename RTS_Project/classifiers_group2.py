import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import rts as util

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = util.data_Xy_split('rts.csv')

    #Leave one out
    knn_cls = KNeighborsClassifier()
    loo = LeaveOneOut()
    parameters = {'n_neighbors':np.arange(2,16)}
    clf = GridSearchCV(knn_cls, parameters,cv=loo)
    clf.fit(X_train,y_train)
    results = np.array(pd.DataFrame(clf.cv_results_)[['param_n_neighbors','mean_test_score','std_test_score']])
    #print(results)
    k_param = results[np.argmax(results[:,1]),0] #find knn-parameter with best CV accuracy score
    util.plot_cv_lambda('KNN_CV.png','KNN CV Results','K Parameter', results[:,0], results[:,1])

    #plot decision boundary with tuned KNN paramter
    knn_clf = KNeighborsClassifier(n_neighbors=k_param)
    knn_clf.fit(X_train,y_train)
    util.plot_decision_boundary(knn_clf,'KNN Classifier', X_train, X_test, y_train, y_test)
    util.plot_ROC_curve(knn_clf,'KNN Classifier', X_test, y_test)
    util.report_accuracy_score(knn_clf, 'KNN Classifier', X_test, y_test)

    #LDA analysis
    lda_clf = LinearDiscriminantAnalysis()
    lda_clf.fit(X_train,y_train)
    util.plot_decision_boundary(lda_clf,'LDA Classifier', X_train, X_test, y_train, y_test)
    util.plot_ROC_curve(lda_clf, 'LDA Classifier', X_test, y_test)
    util.report_accuracy_score(lda_clf, 'LDA Classifier', X_test, y_test)

    #QDA analysis
    qda_clf = QuadraticDiscriminantAnalysis()
    qda_clf.fit(X_train,y_train)
    util.plot_decision_boundary(qda_clf, 'QDA Classifier', X_train, X_test, y_train, y_test)
    util.plot_ROC_curve(qda_clf, 'QDA Classifier', X_test, y_test)
    util.report_accuracy_score(qda_clf, 'QDA Classifier', X_test, y_test)

    #Naive Bayes analysis
    #bayes_clf = GaussianNB(var_smoothing=0.010)
    bayes_clf = GaussianNB()
    bayes_clf.fit(X_train, y_train)
    util.plot_decision_boundary(bayes_clf, 'Naive Bayes Classifier', X_train, X_test, y_train, y_test)
    util.plot_ROC_curve(bayes_clf, 'Naive Bayes Classifier', X_test, y_test)
    util.report_accuracy_score(bayes_clf, 'Naive Bayes Classifier', X_test, y_test)

    pass