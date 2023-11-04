import cv2
import os
import base64
import requests
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# tranmodel = pickle.load(open('carTranModel.pkl','rb'))
# print(len(tranmodel))

Tranmodel = pickle.load(open(r'E:\projectAI\preparedata\model\star_test.pkl', 'rb'))
Tranmodel_np = np.array(Tranmodel)
X_train = Tranmodel_np[:, 0:-1]
y_train = Tranmodel_np[:, -1]


Testmodel = pickle.load(open(r'E:\projectAI\preparedata\model\star_train.pkl', 'rb'))
Testmodel_np = np.array(Testmodel)
X_test = Testmodel_np[:, 0:-1]
y_test = Testmodel_np[:, -1]

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion matrix:", metrics.confusion_matrix(y_test, y_pred))
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

path = r"E:\projectAI\starclass\genstar.pkl"
pickle.dump(clf, open(path, "wb"))
print("data preparation is done")