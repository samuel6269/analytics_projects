<h1 align="center">RTS Noise Detection Project</h1>

This project compares 8 different Machine Learning classification models to determine an optimal model that detects the presence of RTS noise within a sample batch of ICs. A leave one out cross validation technique was implemented for hyperparameter tuning, and the best model was chosen based on the classification precision. Precision was used because of the importance of limiting false positives. In other words, limiting the number of IC products that were labeled passing but contained noise had worse consequences than labeling a good product as a failing unit. The data features were redacted due to company specific data. Here are a description of the project files.

- classifiers_group1.py: This module evaluated the following models: SVM (With Linear and RBF kernel), Classification Decision Tree, and Logistic Regression.

- classifiers_group2.py: This module evaluated the following models: KNN, LDA, QDA, and Naive Bayes

- rts.csv: Contains the raw data results for a batch of products. The data features are removed due to the use of company specific data.
