import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#df = pd.read_csv('First.csv')
#df = pd.read_csv('Territiria12.csv')
#df = pd.read_csv('v75.csv')
df = pd.read_csv('V1185.csv')

#print(df.head())

train = df

######################################################################################
##### TRAINING MODEL #####################
features = list(train.columns[:118])
print(features)
x = train[features]

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# tree = DecisionTreeClassifier(criterion='entropy',
#                               min_samples_leaf=20,
#                               max_leaf_nodes=30,
#                               random_state=20)
tree = DecisionTreeClassifier()
clf = SVC(kernel='linear', probability=True)
########################################################################################

from joblib import dump, load

y = train['Heart1']
clf.fit(x, y)
dump(clf, 'Model1')
print("1 ready")

clf = SVC(kernel='linear', probability=True)
y = train['Heart2']
clf.fit(x, y)
dump(clf, 'Model2')
print("2 ready")

clf = SVC(kernel='linear', probability=True)
y = train['Heart3']
clf.fit(x, y)
dump(clf, 'Model3')
print("3 ready")

clf = SVC(kernel='linear', probability=True)
y = train['Heart4']
clf.fit(x, y)
dump(clf, 'Model4')
print("4 ready")

clf = SVC(kernel='linear', probability=True)
y = train['Heart5']
clf.fit(x, y)
dump(clf, 'Model5')
print("5 ready")

# test = df.tail(int(len(df)*0.3))
#
# features = list(test.columns[:118])
# x = test[features]
#
#
# y_true = test['Heart1']
# y_pred = svcClas1.predict(x)
# print(y_pred)
# print(y_true)
# print("Model1")
# print(accuracy_score(y_true, y_pred))
