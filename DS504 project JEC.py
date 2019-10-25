# Jeffrey E. Crosby
# Fraud Prediction
# DS504 Big Data Analytics
# Worcester Polytechnic Institute
# 10/19/2019 

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz

from time import time

# Algorithms 
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

# Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Pipeline Tools
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

# The main objective is to make a selection of the most precise algorithm, then to classify an input X_prediction

# 1) Data importation
# 2) Class balance
# 3) Import algorithm and cross validation
# 4) Select the best algorithm and make prediction

data = pd.read_csv('C:/data/paysim/PS_20174392719_1491204439457_log.csv')
pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)
print(data.head())
print (50*'=')
print(data.shape)
data.describe()
data.info(null_counts=True, verbose=True)


#----------------------------------------------------
# replace transaction types with integer
# Convert strings to numeric values:
# CASH_IN = 0
# CASH_OUT = 1
# DEBIT = 2
# PAYMENT = 3
# TRANSFER = 4

mapping = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT':3, 'TRANSFER': 4}
data = data.replace({'type': mapping})

# remove letters from nameOrig and nameDest
data['nameOrig'] = data['nameOrig'].map(lambda x: x.lstrip('mMcC'))
data['nameDest'] = data['nameDest'].map(lambda x: x.lstrip('mMcC'))

# plot density of instances by step
p1=sns.kdeplot(data['step'], shade=True, color="steelblue")
plt.show()

# plot number of timepoints at each interval
plt.figure(figsize=(12,4))
sns.distplot(data.step, color='steelblue') #756bb1
plt.title('Time Series')
plt.xlabel('Time Step')
plt.ylabel('Density')
plt.show()

# plot zoomed histograms of amounts
plt.figure(figsize=(10,8))
plt.subplot2grid((2,2), (0,1))
data[data.amount < 10_000_00].amount.hist( color='steelblue',  
                                        edgecolor='white')
plt.axvline(data.amount.mean(),  
            color='r', linestyle='dashed', linewidth=2)
plt.title('Amount Histogram (Zoomed)')
plt.xlabel('Transaction Amount')
plt.show()

#  pie chart of fraud transactions vs. legitimate
labels = 'Fraud', 'Legitimate'
number_fraud_transaction = len(data[data.isFraud == 1])
number_legitimate_transaction = len(data[data.isFraud == 0])
explode = (0.1, 0)
fig1, ax1 = plt.subplots()
ax1.pie([number_fraud_transaction, number_legitimate_transaction],explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()

#  pie chart of flagged fraud transactions vs. unflagged (all)
labels = 'Flagged', 'Unflagged'
number_flagged_transaction = len(data[data.isFlaggedFraud == 1])
number_unflagged_transaction = len(data[data.isFlaggedFraud == 0])
explode = (0.1, 0)
fig1, ax1 = plt.subplots()
ax1.pie([number_flagged_transaction, number_unflagged_transaction],explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()

# evaluate the balance of fraud (1) transactions and legitimate (0)
number_fraud_transaction = len(data[data.isFraud == 1])
number_legitimate_transaction = len(data[data.isFraud == 0])
print('number of fraud transactions: ',number_fraud_transaction)
print(20*'-')
print('number of legitimate transactions: ',number_legitimate_transaction)
print(20*'=')
print("Percentage of fraudulent transactions in the full data set: "+"{:.4%}".format(number_fraud_transaction/number_legitimate_transaction))

# This dataset is poorly balanced at the class level (isFraud).
# Constructing and training a machine learning algorithm on this dataset
# may return class 0 in 99% of cases, and report misleading results. We must
# balance the fraudulent and legitimate transactions in the dataset

# get the index of the legitimate transactions : 
legitimate_transaction_index = data[data.isFraud == 0].index
legitimate_transaction_index = np.array(legitimate_transaction_index)

# get the index of the fraud transactions : 
fraud_transaction_index = data[data.isFraud == 1].index
fraud_transaction_index = np.array(fraud_transaction_index)

# among normal transaction indexes, a dataset of the same size as fraudulent
# transactions is randomly selected
random_legitimate_transaction_index = np.random.choice(legitimate_transaction_index,number_fraud_transaction, replace = False)
random_legitimate_transaction_index = np.array(random_legitimate_transaction_index)

# we have our index selection, we have to group the two samples
selection_index = np.concatenate([fraud_transaction_index,random_legitimate_transaction_index])

# and we recover the data associated with these indexes
selected_data = data.iloc[selection_index,:]

# checking the size
print( pd.DataFrame ( {'NB' : selected_data.groupby(['isFraud']).size()}).reset_index()) 

# plot histograms of selected data
npsdata = np.array(selected_data)
npsdata = npsdata[:,[0,1,2,4,5,7,8,9,10]]
npsdcolumns = data.columns[[0,1,2,4,5,7,8,9,10]]

fig, axes = plt.subplots(5, 2, figsize=(10, 8))
legitimate = npsdata[npsdata[:,7] == 0]
fraud = npsdata[npsdata[:,7] == 1]
ax = axes.ravel()
for i in range(9):
	_, bins = np.histogram(npsdata[:, i], bins = 50)
	ax[i].hist(fraud[:, i], bins=bins, color='red', alpha=.5)
	ax[i].hist(legitimate[:, i], bins=bins, color='steelblue', alpha=.5)
	ax[i].set_title(npsdcolumns[i])
	ax[i].set_yticks(())
	ax[0].set_xlabel("feature magnitude")
	ax[0].set_ylabel("frequency")
	ax[0].legend(["fraud","legitimate"], loc="best")
	fig.tight_layout()
plt.show()

# ======================================================================

# the data is balanced. we can now prepare the data for training and testing
X = selected_data.drop(['isFraud','isFlaggedFraud'], axis = 1)
y = selected_data[['isFraud']]

# split into training and test data sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)

# X_train = np.array(X_train) 
# y_train = np.array(y_train)

# further split test into test and validation sets

X_test, X_val, y_test, y_val = train_test_split(X_test,y_test, test_size=0.5, random_state=1)


# evaluation model
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(y_test, [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test, [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(y_test, predictions)
    results['precision'] = precision_score(y_test, predictions)
    results['roc'] = roc_auc_score(y_test, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(y_train, train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');

# display a DecisionTreeClassifier
# -----------------------------------------------------
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
tree.plot_tree(clf.fit(X_train,y_train))

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
		filled = True, rounded = True,
		special_characters = True,
                feature_names = selected_data.columns.tolist()[0:9],
                class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('fraud.png')
Image(graph.create_png())
plt.show()

#=======================================================
# implement Training RandomForestClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
classifier = RandomForestClassifier(n_estimators=100,
                                    random_state=0,
                                    max_features = 'sqrt',
                                    n_jobs = -1,
                                    verbose = 1)
classifier.fit(X_train,y_train)
n_nodes = []
max_depths = []

for ind_tree in classifier.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

average_precision = average_precision_score(y_test, y_pred)

y_pred = classifier.predict(X_train) 
precision, recall, thresholds = precision_recall_curve(y_train, y_pred)
average_precision = average_precision_score(y_train, y_pred)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Training Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
plt.show()

# implement Validation RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,
                                    random_state=0,
                                    max_features = 'sqrt',
                                    n_jobs = -1,
                                    verbose = 1)
classifier.fit(X_val,y_val)
n_nodes = []
max_depths = []

for ind_tree in classifier.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

y_pred = classifier.predict(X_val) 
precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
average_precision = average_precision_score(y_val, y_pred)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Validation Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
plt.show()

# implement Testing RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,
                                    random_state=0,
                                    max_features = 'sqrt',
                                    n_jobs = -1,
                                    verbose = 1)
classifier.fit(X_test,y_test)
n_nodes = []
max_depths = []

for ind_tree in classifier.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

y_pred = classifier.predict(X_test) 
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Testing Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
plt.show()

#===============================================================
# define the ROC curves

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

# fit a model on the training data 
model = RandomForestClassifier()
model.fit(X_train, y_train)

# predict the probabilities for the test data
probs = model.predict_proba(X_test)

# keep the probabilities for the positive class only
probs = probs[:, 1]

# compute the AUC score
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)

# get the ROC curve 
fpr, tpr, thresholds = roc_curve(y_test, probs)

# plot the ROC curve using the defined function
plot_roc_curve(fpr, tpr)


# ==================================================================
# create Precision- Recall Curve - Training

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
classifier = RandomForestClassifier(n_estimators=100,
                                    random_state=0,
                                    max_features = 'sqrt',
                                    n_jobs = -1,
                                    verbose = 1)
classifier.fit(X_train,y_train)
n_nodes = []
max_depths = []

for ind_tree in classifier.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

y_pred = classifier.predict(X_train) 
precision, recall, thresholds = precision_recall_curve(y_train, y_pred)
average_precision = average_precision_score(y_train, y_pred)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Training Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
plt.show()



# ===============================================================
# IsolationForest

to_model_columns=X_train.columns[0:9]
# metrics_df=X_train.iloc[:,0:9]
clf=IsolationForest(n_estimators=100,
                    max_samples='auto',
                    contamination=float(.12),
                    max_features=1.0,
                    bootstrap=False,
                    n_jobs=-1,
                    random_state=42,
                    verbose=0)
clf.fit(X_train[to_model_columns])
pred = clf.predict(X_train[to_model_columns])
X_train['anomaly']=pred
outliers=X_train.loc[X_train['anomaly']==-1]
outlier_index=list(outliers.index)

print(X_train['anomaly'].value_counts())

# normalize and fit the metrics to a PCA and plot
# pca = PCA(n_components=3)  # reduce to k=3 dimensions
# scaler = StandardScaler()
# normalize the metrics
# X = scaler.fit_transform(metrics_df[to_model_columns]) # lose the index here, and previous X of 16426 rows
# X = pca.fit(metrics_df[to_model_columns])

# add back in the index values from the original data set
# X_reduce = np.hstack((X_reduce, np.atleast_2d(metrics_df.index.values).T))

# ================================================================================
# pca = PCA(n_components=3)  # reduce to k=3 dimensions
# X_train = pca.fit_transform(X_train)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_zlabel("x_composite_3")

# plot the compressed data points
# ax.scatter(X_train[:, 0], X_train[:, 1], zs=X_train[:, 2], s=4, lw=1, label="inliers",c="green")

# plot xs for the ground truth outliers
# ax.scatter(X_train[outlier_index,0],X_train[outlier_index,1], X_train[outlier_index,2],
#           lw=2, s=60, marker="x", c="red", label="outliers")

# ax.legend()
# plt.show()


# --------------------
# index_val = pd.Index(selected_data.index).to_list()
# selected_data['index_val'] = index_val


# pca = PCA(n_components = 5)
# pca.fit(metrics_df)

# -----------------------
# pca = PCA(n_components=3)
# principalComponents = pca.fit_transform(X_train)
# principalDf = pd.DataFrame(data = principalComponents,
			   # columns = ['principal component 1',
				#      'principal component 2',
				#      'principal component 3'])
# finalDf = pd.concat([principalDf, y_train[['isFraud']]], axis = 1)

# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Prinicapal Component 1', fontsize = 15)
# ax.set_ylabel('Prinicapal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize=20)

# targets = [0,1]
# colors = ['r','g','b']
# for target, color in zip(targets,colors):
  #  indicesToKeep = finalDf['target'] == target
  #  ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
  #             finalDf.loc[indicesToKeep, 'principal component 2'],
  #             c = color,
  #             s = 50)
# ax.legend(targets)
# ax.grid()


# ========================================================================================
# testing of multiple machine learning methods/algoritms


# Import algorithms 
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),  
    ]

# Algorithm training and cross validation

# Dataframe to compare algorithms
col = ['MLA Name','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Time']
MLA_compare = pd.DataFrame(columns = col)

# Cross validation split
cv_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .2, train_size = .8, random_state = 0)


index = 0 
for alg in MLA: 
    # Name of algorithm
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[index, 'MLA Name'] = MLA_name
    
    # Cross validation
    cv_results = model_selection.cross_validate(alg, X, y, cv  = cv_split, return_train_score=True)
    MLA_compare.loc[index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    MLA_compare.loc[index,'MLA Time'] = cv_results['fit_time'].mean()

    index +=1

# Print comparison table
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
print(MLA_compare)

sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'blue')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')

plt.show()

# We will select the best algorithm to make future predictions

best_score = max(MLA_compare['MLA Test Accuracy Mean'])
best_MLA = MLA_compare[MLA_compare['MLA Test Accuracy Mean'] == best_score].reset_index(drop = True)
name_best_alg = np.array(best_MLA['MLA Name'])
name_best_alg = name_best_alg[0]
print("The best algorithm is:",name_best_alg)






