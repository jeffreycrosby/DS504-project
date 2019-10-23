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

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve
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
print(data.head())
print (50*'=')
print(data.shape)
data.describe()
data.info(null_counts=True, verbose=True)


# graph data
plt.plot(data.step)
plt.show()

plt.plot(data.step,data.amount) # plot amount on y axis, step (hour) on x axis
plt.show()

plt.plot(data.step,data.oldbalanceOrg)
plt.show()

plt.plot(data.step,data.newbalanceOrig)
plt.show()


# the oldbalanceDest plot shows an interesting shift from around record 3,600,000 on
plt.plot(data.oldbalanceDest)
plt.show()

# the oldbalanceDest plotted agains the step (hour) over the course of the month captured

plt.plot(data.step, data.oldbalanceDest)
plt.show()

# plot histograms of initial data
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


npdata = np.array(data)
npdata = npdata[:,[0,1,2,4,5,7,8,9,10]]
npdcolumns = data.columns[[0,1,2,4,5,7,8,9,10]]

fig, axes = plt.subplots(5, 2, figsize=(10, 8))
legitimate = npdata[npdata[:,7] == 0]
fraud = npdata[npdata[:,7] == 1]
ax = axes.ravel()
for i in range(9):
	_, bins = np.histogram(npdata[:, i], bins = 50)
	ax[i].hist(fraud[:, i], bins=bins, color='red', alpha=.5)
	ax[i].hist(legitimate[:, i], bins=bins, color='steelblue', alpha=.5)
	ax[i].set_title(npdcolumns[i])
	ax[i].set_yticks(())
	ax[0].set_xlabel("feature magnitude")
	ax[0].set_ylabel("frequency")
	ax[0].legend(["fraud","legitimate"], loc="best")
	fig.tight_layout()
plt.show()
	
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

# plot histogram of amounts
plt.figure(figsize=(10,8))
plt.subplot2grid((2,2),(0,0))
data.amount.hist(color='steelblue', edgecolor='white')
plt.axvline(data.amount.mean(),
            color='r',linestyle='dashed',linewidth=2)
plt.title('Amount Histogram')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')

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

# print the number of flagged transactions

print('number of fraud transactions:',number_fraud_transaction)
print('number of legitimate transactions:',number_legitimate_transaction)
print('number of flagged transactions:',number_flagged_transaction)
print('number of unflagged transactions:',number_unflagged_transaction)

# We must first look at the balancing between class 1 (fraudulent transaction), and class 0 (normal transaction).

number_fraud_transaction = len(data[data.isFraud == 1])
number_legitimate_transaction = len(data[data.isFraud == 0])

print('number of fraud transactions: ',number_fraud_transaction)
print(20*'-')
print('number of legitimate transactions: ',number_legitimate_transaction)

# Unfortunately, our dataset is very poorly balanced at the class level. Constructing and training a machine learning
# algorithm on this dataset may return class 0 in 99% of cases. It's misleading. We must rebalance the dataset

# get the index of the legitimate transactions : 
legitimate_transaction_index = data[data.isFraud == 0].index
legitimate_transaction_index = np.array(legitimate_transaction_index)
# get the index of the fraud transactions : 
fraud_transaction_index = data[data.isFraud == 1].index
fraud_transaction_index = np.array(fraud_transaction_index)


# Among normal transaction indexes, a dataset of the same size as fraudulent transactions is randomly selected
random_legitimate_transaction_index = np.random.choice(legitimate_transaction_index,number_fraud_transaction, replace = False)
random_legitimate_transaction_index = np.array(random_legitimate_transaction_index)

# We have our index selection, we have to group the two samples
selection_index = np.concatenate([fraud_transaction_index,random_legitimate_transaction_index])

# And we recover the data associated with these indexes
selected_data = data.iloc[selection_index,:]

#Checking the size
print( pd.DataFrame ( {'NB' : selected_data.groupby(['isFraud']).size()}).reset_index()) 


# remove letters from nameOrig and nameDest

selected_data['nameOrig'] = selected_data['nameOrig'].map(lambda x: x.lstrip('mMcC'))
selected_data['nameDest'] = selected_data['nameDest'].map(lambda x: x.lstrip('mMcC'))

# perform on the full data set as well

data['nameOrig'] = data['nameOrig'].map(lambda x: x.lstrip('mMcC'))
data['nameDest'] = data['nameDest'].map(lambda x: x.lstrip('mMcC'))

# replace transaction types with integer
# Convert strings to numeric values:
# CASH_IN = 0
# CASH_OUT = 1
# DEBIT = 2
# PAYMENT = 3
# TRANSFER = 4

mapping = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT':3, 'TRANSFER': 4}
selected_data = selected_data.replace({'type': mapping})

# plot histograms of selected data
#
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


             
# The data is balanced. we can now prepare the data for training and testing
X = selected_data.drop(['isFraud','isFlaggedFraud'], axis = 1)
y = selected_data[['isFraud']]

# split into training and test data sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)

X_train = np.array(X_train)
y_train = np.array(y_train)

# further split test into test and validation sets

X_test, X_val, y_test, y_val = train_test_split(X_test,y_test, test_size=0.5, random_state=1)

X_test = np.array(X_test)
X_val = np.array(X_val)
y_test = np.array(y_test)
y_val = np.array(y_val)

#---------------------------------------------------------------------------------
# perform drop and column selection for full data also
X_full = data.drop(['isFraud','isFlaggedFraud'], axis = 1)
y_full = data[['isFraud']]

# split into training and test data sets

X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(X_full,y_full, test_size=0.3, random_state=1)

X_full_train = np.array(X_full_train)
y_full_train = np.array(y_full_train)

# further split test into test and validation sets

X_full_test, X_full_val, y_full_test, y_full_val = train_test_split(X_full_test,y_full_test, test_size=0.5, random_state=1)

X_full_test = np.array(X_full_test)
X_full_val = np.array(X_full_val)
y_full_test = np.array(y_full_test)
y_full_val = np.array(y_full_val)

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

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
tree.plot_tree(clf.fit(X,y))

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
		filled = True, rounded = True,
		special_characters = True,
                feature_names = selected_data.columns.tolist()[0:10],
                class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('fraud.png')
Image(graph.create_png())
plt.show()

#=======================================================
# implement Training RandomForestClassifier 
from sklearn.ensemble import RandomForestClassifier
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



#===============================================================



# create ROC curve
train_rf_predictions = classifier.predict(X_train)
train_rf_probs = model.predict_proba(X_train)[:, 1]

rf_predictions = classifier.predict(X_test)
rf_probs = classifier.predict_proba(X_test)[:, 1]

evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)
plt.show()

# -----------------------------------------------
# calculate precision-recall curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from inspect import signature

precision, recall, threshold = precision_recall_curve(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred)

step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
plt.step(recall, precision, color='r', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='steelblue', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()

# calculate the f1 score
from sklearn.metrics import f1_score
y_pred = classifier.predict(X_test) 
# yhat = classifier.predict(X_test) # predict class values
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, yhat)

# calculate precision-recall AUC
from sklearn.metrics import auc
auc = auc(recall, precision)


# generate a no skill prediction (majority class)
no_skill_probs = [0 for _ in range(len(y_test))]

# calculate precision and recall for each threshold
ns_precision, ns_recall, _ = precision_recall_curve(y_test, no_skill_probs)
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probs)

# calculate scores
ns_f1, ns_auc = f1_score(y_test, no_skill_probs), auc(ns_recall, ns_precision)
rf_f1, rf_auc = f1_score(y_test, yhat), auc(rf_recall, rf_precision)

# summarize scores
print('No Skill: f1=%.3f auc=%.3f' % (ns_f1, ns_auc))
print('Random Forest: f1=%.3f auc=%.3f' % (rf_f1, rf_auc))

# plot the precision-recall curves
plt.plot(ns_recall, ns_precision, linestyle='--', label='No Skill')
plt.plot(rf_recall, rf_precision, marker='.', label='Random Forest')

# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')

# show the legend
plt.legend()

# show the plot
plt.show()
# -----------------------------------------------------
# basic PRC model
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Training Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
plt.show()

# ------------------------------------------------------


y_pred=classifier.predict(X_test)
classifier.score(X_test,y_test)
cnf_matrix = confusion_matrix(y_test,y_pred)












print('Mean Absolute Error:', metrics.mean_absolute_error(y,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
print('Root Mean Squared Error:',metrics.mean_squared_error(y,y_pred))

print(confusion_matrix(y,y_pred))
print(classification_report(y,y_pred))
print(accuracy_score(y, y_pred))

# plot ROC curve for the full data set
fpr, tpr, thresholds = metrics.roc_curve(y_full_test,y_full_pred)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for fraud classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

print('Area Under the Curve (AUC) score:',metrics.roc_auc_score(y_full_test, y_full_pred))


# implement RandomForestClassifier on full data set

classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_full_train,y_full_train)
y_full_pred = classifier.predict(X_full_test)

print('Mean Absolute Error (full data set):', metrics.mean_absolute_error(y_full_test,y_full_pred))
print('Mean Squared Error (full data set):', metrics.mean_squared_error(y_full_test, y_full_pred))
print('Root Mean Squared Error (full data set):',metrics.mean_squared_error(y_full_test,y_full_pred))

print(confusion_matrix(y_full_test,y_full_pred))
print(classification_report(y_full_test,y_full_pred))
print(accuracy_score(y_full_test, y_full_pred))


# plot ROC curve for the full data set
fpr, tpr, thresholds = metrics.roc_curve(y_full_test,y_full_pred)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for fraud classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

print('Area Under the Curve (AUC) score:',metrics.roc_auc_score(y_full_test, y_full_pred))







