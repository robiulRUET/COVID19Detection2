"""Import all the necessary packages"""
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


"""Test for GPU or CPU devices"""
print("Device: \n", tf.config.experimental.list_physical_devices())
print(tf.__version__)
print(tf.test.is_built_with_cuda())

"""Contrast Limited Adaptive Histogram Equalization (CLAHE) Method for preprocessing"""

def clahe_function(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab_img)
    clahe_img = clahe.apply(lab_img[0])
    updated_lab_img2 = cv2.merge(lab_planes)
    # Convert LAB image back to color (RGB)
    CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
    return CLAHE_img

"""Funcntion for loading images"""

images = []
def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        #img = cv2.resize(img, (224, 224))
        img = clahe_function(img)
        img = img/255.0
        img = cv2.resize(img, (224, 224))
        if img is not None:
            images.append(img)
    return images

covid = r'F:/Covid-19 Ct scan/Kaggle DATA/Data/COVID/'   # directory of covid image
non_covid = r'F:/Covid-19 Ct scan/Kaggle DATA/Data/non-COVID/'   # directory of non-covid image

load_images_from_folder(covid)
load_images_from_folder(non_covid)

y = np.ones(1229)  #labeling non-covid images as 1
y = np.append(y, np.zeros(1252))   #labeling covid images as 0
y = list(y)
c = list(zip(images, y))

#reshuffling all the images along with their labels

random.shuffle(c)
images, y = zip(*c)
del c  #For Memory Efficiency
images = np.array(images)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.10, stratify= y, random_state =2)

"""Our proposed Convolutional Model for extracting features"""

model = tf.keras.Sequential()
#1st conv layer
model.add(tf.keras.layers.Conv2D(8, 3, input_shape=(224, 224, 3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(2))
#2nd conv layer
model.add(tf.keras.layers.Conv2D(16, 3, padding="valid"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(2))
#3rd conv layer (from here model gives good result)
model.add(tf.keras.layers.Conv2D(32, 3, padding="valid"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(2))
#4th conv layer
model.add(tf.keras.layers.Conv2D(64, 3, padding="valid"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(2))
#Flatten Layer
model.add(tf.keras.layers.Flatten())
#Dense Layer 1
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dropout(0.5))
#Dense Layer 2
model.add(tf.keras.layers.Dense(100,name ='feature_denseee')) #100 Prominant Features are Extracted From This Layer
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dropout(0.5))
#output Dense Layer
model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.Activation('softmax'))
adam = tf.keras.optimizers.Adam(lr=0.001)

model.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')
model.summary()

#CNN Model Trained
with tf.device('/GPU:0'):
    history = model.fit(X_train, y_train, batch_size= 32, epochs=100, verbose=1, validation_data=(X_test, y_test))

""" Save the model """
model.save("F:/Pneumonia/model/covid.h5")

""" Load the model """
model = tf.keras.models.load_model('F:/Pneumonia/model/covid.h5')
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('feature_denseee').output)
intermediate_layer_model.summary()

"""Features Extraction From All Images"""
feature_engg_data = intermediate_layer_model.predict(images)
feature_engg_data = pd.DataFrame(feature_engg_data)

"""Save The Features"""
feature_engg_data.to_pickle('F:/Pneumonia/model/finalfeaturescovid.pkl')
features = pd.read_pickle('F:/Pneumonia/model/finalfeaturescovid.pkl')

"""Normalization of The Features"""
from sklearn.preprocessing import StandardScaler
x = feature_engg_data.loc[:, feature_engg_data.columns].values
x = StandardScaler().fit_transform(x)

"""Splitting the Data into Training & Testing Set"""
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, stratify= y, random_state =0)

"""Apply The Machine Learning ALgorithms For Classification"""

"""Gaussian Naive Bayes"""

from sklearn.naive_bayes import GaussianNB
from sklearn import  metrics
model = GaussianNB()
model.fit(X_train, y_train)
y_pred1 = model.predict(X_test)
metrics.accuracy_score(y_test, y_pred1)
print(metrics.roc_curve(y_test, y_pred1))
print(metrics.confusion_matrix(y_test, y_pred1))
print(metrics.classification_report(y_test, y_pred1))

"""Support Vector Machine"""

from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid', probability=True)
svclassifier.fit(X_train, y_train)
y_pred2 = svclassifier.predict(X_test)
metrics.accuracy_score(y_test, y_pred2)
metrics.plot_roc_curve(svclassifier, X_test, y_test)
metrics.plot_confusion_matrix(svclassifier, X_test, y_test)
print(metrics.confusion_matrix(y_test, y_pred2))
print(metrics.classification_report(y_test, y_pred2))

"""Decision Tree Classifier"""

from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)
y_pred3 = dt.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred3))
metrics.roc_curve(y_test, y_pred3)
metrics.plot_roc_curve(dt, X_test, y_test)
metrics.plot_confusion_matrix(dt, X_test, y_test)
print(metrics.confusion_matrix(y_test, y_pred3))
print(metrics.classification_report(y_test, y_pred3))

"""Logistic Regression"""

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=1)
lr = lr.fit(X_train, y_train)
y_pred4 = lr.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred4))
metrics.plot_roc_curve(lr, X_test, y_test)
metrics.plot_confusion_matrix(lr, X_test, y_test)
print(metrics.confusion_matrix(y_test, y_pred4))
print(metrics.classification_report(y_test, y_pred4))

"""Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=50, random_state=1)
RF = RF.fit(X_train, y_train)
y_pred5 = RF.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred5))
metrics.plot_confusion_matrix(RF, X_test, y_pred5)
print(metrics.classification_report(y_test, y_pred5))

"""Ensemble Model"""

from sklearn.ensemble import VotingClassifier
voting_classifier = VotingClassifier(estimators=[('GNB', model), ('svm', svclassifier), ('dt', dt), ('lr', lr), ('rf', RF)], voting='hard')
voting_classifier.fit(X_train, y_train)
y_pred_vot = voting_classifier.predict(X_test)
metrics.accuracy_score(y_test, y_pred_vot)
metrics.plot_confusion_matrix(voting_classifier, X_test, y_pred_vot)
print(metrics.classification_report(y_test, y_pred_vot))

"""Plotting Various Matrix"""

plt.figure(0).clf()

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred1)
auc = metrics.roc_auc_score(y_test, y_pred1)
plt.plot(fpr,tpr,label="GNB, auc="+str('{0:.4f}'.format(auc)))


fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred2)
auc = metrics.roc_auc_score(y_test, y_pred2)
plt.plot(fpr,tpr,label="SVM, auc="+str('{0:.4f}'.format(auc)))

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred3)
auc = metrics.roc_auc_score(y_test, y_pred3)
plt.plot(fpr,tpr,label="DT, auc="+str('{0:.4f}'.format(auc)))

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred4)
auc = metrics.roc_auc_score(y_test, y_pred4)
plt.plot(fpr,tpr,label="LR, auc="+str('{0:.4f}'.format(auc)))

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred5)
auc = metrics.roc_auc_score(y_test, y_pred5)
plt.plot(fpr,tpr,label="RF, auc="+str('{0:.4f}'.format(auc)))

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred_vot)
auc = metrics.roc_auc_score(y_test, y_pred_vot)
plt.plot(fpr,tpr,label="Ensembled Model, auc="+str('{0:.4f}'.format(auc)))

plt.title("models performance")
plt.xlabel("1-Specificity(False Positive Rate)")
plt.ylabel("Sensitivity(True Positive Rate)")
plt.legend(loc=0)