# Data-Detection-ML-Fundamentals
Deep Learning backed by Supervised and Unsupervised PCA analysis of Tabular Medical Data with MIMIC-III using Advanced visuallization methods.
workflow: Raw Data interpretation -> ML models -> Clustering -> Deep Neural Networks

Step 1: understanding raw data. examining the most cruciel elemnts of the data.
<img width="1124" height="597" alt="image" src="https://github.com/user-attachments/assets/43187d4a-0f42-4214-a87e-d7edf6ddb7e9" />

check the relation between BMI and age:

<img width="446" height="443" alt="image" src="https://github.com/user-attachments/assets/9ba83a9a-e351-4932-9708-9847b5bb7860" />

Step 2: Train ML models to find patterns and understand the data:
Model Performance Summary:

                model  accuracy  precision  recall    f1
0  LogisticRegression     0.855      0.000   0.000  0.00

1                 SVM     0.855      0.000   0.000  0.00

2                 KNN     0.839      0.000   0.000  0.00

3        DecisionTree     0.793      0.167   0.107  0.13

 ----------------------------------------------------------------- 


 Best Model: LogisticRegression

              precision    recall  f1-score   support

         0.0       0.85      1.00      0.92       165
         
         1.0       0.00      0.00      0.00        28

    accuracy                           0.85       193
   macro avg       0.43      0.50      0.46       193
weighted avg       0.73      0.85      0.79       193

<img width="292" height="289" alt="image" src="https://github.com/user-attachments/assets/80379185-409c-4cee-b02e-68e5770863d3" />


 Confusion Matrix Explanation:
 TN: 165 - Correctly predicted class 0
 FP: 0 - Predicted 1 but it was 0
 FN: 28 - Predicted 0 but it was 1
 TP: 0 - Correctly predicted class 1

Predict BMI based on age and Blood sodium with linear regression, SVM regressor, Decision tree regressor, and kNN refressor.
                       MSE    RMSE     R²
Linear Regression   63.689   7.981 -0.023
SVM Regressor       58.340   7.638  0.063
Decision Tree      116.781  10.807 -0.876
kNN Regressor       62.333   7.895 -0.001

<img width="795" height="662" alt="image" src="https://github.com/user-attachments/assets/8945ced5-fdff-45f2-8378-313d256524f8" />

Step 3: PCA and t-SNE

Demonstrate the application of Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction on the dataset focusing on BMI, Blood sodium, and Blood calcium to visualize the data in a reduced-dimensional space. Compare the visualization results of PCA and t-SNE.

<img width="599" height="420" alt="image" src="https://github.com/user-attachments/assets/8322a1a4-d5d1-4d0c-b61c-45f311f0719e" />

Explained variance ratio: [0.84059511 0.15633159]
Principal component direction(s):
[[ 0.99960957  0.02731272  0.00589272]
 [-0.02736161  0.99959047  0.00838121]]

Differences between Methods data manipulations:
<img width="1106" height="466" alt="image" src="https://github.com/user-attachments/assets/64cdd869-3a51-4a5a-9f08-e49753363db7" />

Step 4: Clustering using K-means
Visualization helps to understand the relation between clusters after dimension reduction
PCA clusters:

<img width="1189" height="607" alt="image" src="https://github.com/user-attachments/assets/5d662b9b-23fe-427a-8c23-45c8024797fc" />

t-SNE clusters:

<img width="1185" height="574" alt="image" src="https://github.com/user-attachments/assets/de94a75c-c702-4975-8686-ea173de9e1a2" />

Step 5: Deep Neural Networks 
Training TensorFlow Deep Neural network.

<img width="811" height="372" alt="image" src="https://github.com/user-attachments/assets/5731f296-a24b-4908-b4ea-a05daf416bac" />

Test Accuracy: 0.8691

Conclusion: Dataset is too small to effectivly train a DNN, its similar results to classic ML classification methods like LogisticRegression and SVM.Even when using earlyStopping and advanced methods and tuning, a denser structure and feature list, there is not a big improvement.
