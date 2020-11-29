from joblib import dump, load
import pandas as pd
from sklearn.metrics import accuracy_score

clf1 = load('Model1')
clf2 = load('Model2')
clf3 = load('Model3')
clf4 = load('Model4')
clf5 = load('Model5')

df = pd.read_csv('V1185.csv')

#print(df)

#train = df
test = df.tail(int(len(df)*0.3))

features = list(test.columns[:118])
x = test[features]


y_true = test['Heart1']
y_pred = clf1.predict(x)
# print(y_pred)
# print(y_true)
print("Model1")
print(accuracy_score(y_true, y_pred))

y_true = test['Heart2']
y_pred = clf2.predict(x)
# print(y_pred)
# print(y_true)
print("Model2")
print(accuracy_score(y_true, y_pred))

y_true = test['Heart3']
y_pred = clf3.predict(x)
print("Model3")
print(accuracy_score(y_true, y_pred))

y_true = test['Heart4']
y_pred = clf4.predict(x)
print("Model4")
print(accuracy_score(y_true, y_pred))

y_true = test['Heart5']
y_pred = clf5.predict(x)
print("Model5")
print(accuracy_score(y_true, y_pred))