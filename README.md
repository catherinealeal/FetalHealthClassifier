# Predicting Fetal Health with Tree-Based Models 

## Introduction + Goal

A cardiotocogram (CTG) is a medical test that records a fetus’s heart rate and the uterine contractions of the mother. Features extracted from a CTG often include things like baseline heart rate, variability, accelerations, and decelerations. Doctors use CTG data to assess fetal well-being. The CTG data, along with the doctor’s assessment of the fetus’s health, can also be used to train predictive models for fetal health classification.

The goal of this project is to train three classification models to predict whether a fetus’s health is normal, suspect or pathological based on CTG data. Model performance will be evaluated and the models compared based on accuracy, F1-score, and interpretability. All three classification methods used will be tree-based: a simple Decision Tree, AdaBoost, and Random Forest. A Decision Tree is particularly useful in this case because it can capture non-linear relationships among the cardiotocogram features and provide an interpretable structure for understanding the predictions.

## Data Description + Preprocessing

The features used for model training were automatically extracted from cardiotocogram readings. Expert obstetricians also evaluated each CTG and classified the fetus’s health as normal, suspect, or pathological. Each of the 2126 rows represents a CTG for a mother–fetus pair. 21 columns contain  features extracted from the CTG and the 22nd column contains the experts’ classifications.

Learn more about the dataset [here](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification).

## Model Building

I'm going to build 3 classifiers using different algorithms: decision tree, AdaBoost, and random forest. 

### Decision Tree
A decision tree is a model that makes predictions by recursively splitting data on feature values. Each internal node represents a decision based on a feature (like whether a measurement exceeds a threshold) and each leaf node represents a predicted outcome. Before I can train the model, I'm going to tune its hyperparameters using cross-validation. Specifically, GridSearchCV performs cross-validation for each combination of hyperparameters and selects the best combination based on a performance metric. 

Best parameters: {'max_depth': None, 'min_samples_leaf': 5, 'min_samples_split': 10}

### AdaBoost 
Boosting is a technique to improve the performance of weak classifiers by iteratively reweighting the training data to focus on misclassified points. AdaBoost combines many weak classifiers (shallow decision trees), adjusting their weights at each step. The final classifier aggregates all of the weak learners through weighted majority voting. Before training the model, I will tune its hyperparameters with GridSearchCV to find the best number of estimators and learning rate.

Best parameters: {'learning_rate': 1, 'n_estimators': 150}

### Random Forest

Random Forests is an algorithm that uses bagging (bootstrap aggregating): training many individual decision trees on different random subsets of the data (bootstrapped samples) and random subsets of features. Each tree makes its own prediction, and the final classification is determined by majority vote across all trees. I am again using GridSearchCV to tune the key hyperparameters of the random forest model.

Best parameters: {'max_depth': 15, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400}

## Model Evaluations

I'm going to evaluate and compare the performaces of the 3 classifiers using generalization error and macro F1. 

Generalization error quantifies how well the model will classify unseen data. It is computed as the accuracy score of the model predictions for the held-out testing set. While accuracy provides some measure of model performance, if the classes are imbalanced as they are in this case, accuracy can be misleading. 

Our dataset has more than 3x as many normal health diagnosis examples as suspect and pathological examples combined, so a poor model could classifier every new case as healthy and still have accuracy over 75% despite also having a false negative rate of 100%. 

A metric which accounts for unbalanced classes is the macro F1-score. Generally, an F1-score balances precision and recall to punish models that either miss too many true cases (low recall) or produce too many false positives (low precision). A high F1-score indicates a model is good at classifying true cases without causing too many false positives. A macro F1-score is used for multi-class cases because it computes the F1-score separately for each class and then averages those three scores. This method works better than accuracy for imbalanced classes because it treats all classes equally, regardless of how many examples they have. In this case, macro F1 will emphasize how well the model identifies suspect and pathological fetuses, not just the majority healthy class. 

![image](https://github.com/catherinealeal/FetalHealthClassifier/blob/main/images/table.png)

![image](https://github.com/catherinealeal/FetalHealthClassifier/blob/main/images/plot1.png)

## Analysis 
The Random Forest model achieved the best overall performance, with an accuracy of 0.927 and a macro F1-score of 0.861, outperforming both the single Decision Tree and AdaBoost. The Decision Tree also performed well (accuracy 0.908, F1_macro 0.833), demonstrating that a single interpretable tree can capture much of the structure in the CTG dataset. AdaBoost, on the other hand, performed significantly worse (accuracy 0.798, F1_macro 0.419), likely due to the small dataset size or sensitivity to noisy cases. 

These results highlight that ensembles can improve robustness: Random Forest reduces variance through bagging and generalizes better than a single tree. In terms of interpretability, the single Decision Tree is the easiest to visualize and explain, making it useful for clinical contexts, while Random Forest and AdaBoost are harder to interpret directly. However, both ensembles allow for analysis of feature importance, which can provide insight into the most influential CTG features for making predictions.

![image](https://github.com/catherinealeal/FetalHealthClassifier/blob/main/images/plot2.png)

The Random Forest model identified several CTG features as particularly important for predicting fetal health. The most influential features were abnormal_short_term_variability, percentage_of_time_with_abnormal_long_term_variability, and mean_value_of_short_term_variability, indicating that both short-term and long-term variability metrics are key indicators of fetal well-being. 

## Conclusion 

Altogether, the Random Forest model not only demonstrated the highest predictive performance among the three classifiers but also provides a valuable tool for real-world fetal health assessment. By accurately identifying cases as normal, suspect, or pathological, the model could assist obstetricians in prioritizing high-risk pregnancies and supporting timely clinical interventions. The ability to quantify feature importance further enhances its practical utility, allowing clinicians to understand which aspects of a CTG most strongly influence predictions. While not as immediately interpretable as a single Decision Tree, the combination of high accuracy, robustness to variance, and insight into key features makes Random Forest a promising model for aiding decision-making in prenatal care.

View the full project with code [here](https://github.com/catherinealeal/FetalHealthClassifier/blob/main/FetusHealthClassifier.ipynb).
