# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 22:39:47 2022

@author: carlo
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

negativos = [1] * 60
positivos = [2] * 60

negativosP = [1] * 61
positivosP = [2] * 53
positivos2 = [2] * 6

y_true = negativos + positivos
y_pred = positivos2 + negativosP + positivosP

confm = confusion_matrix(y_true, y_pred)
print(confm)


print()


accuracy = accuracy_score(y_true, y_pred)
print ('El accuracy es: ', accuracy)

#Metrica Precisión

#Para casos positivos
positivo = precision_score(y_true, y_pred, pos_label = 2) 
print ('La precisión : ', positivo)
 

#negativo = precision_score(y_true, y_pred, pos_label = 1) 
#print ('La precisión en los casos negativos es: ', negativo) 

#Metrica Recall

recaall = recall_score(y_true, y_pred)
print('El Recall es: ',recaall)

#MEtrica F-1 Score

f1 = f1_score(y_true, y_pred, average='macro')
print('El F1-Score es : ',f1)

#f2 = f1_score(y_true, y_pred)
#print('El F1-Score con el desbalanceo de clases : ',f2)

#Metrica ROC 
# Generamos un dataset de dos clases
x, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
# Dividimos en training y test
trainX, testX, trainy, testy = train_test_split(x, y, test_size=0.5, random_state=2)
#Generamos un clasificador sin entrenar , que asignará 0 a todo
ns_probs = [0 for _ in range(len(testy))]
# Entrenamos nuestro modelo de reg log
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# Predecimos las probabilidades
lr_probs = model.predict_proba(testX)
#Nos quedamos con las probabilidades de la clase positiva (la probabilidad de 1)
lr_probs = lr_probs[:, 1]
# Calculamos el AUC
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# Imprimimos en pantalla
#print('Sin entrenar: ROC AUC=%.3f' % (ns_auc))
print('AUC=%.3f' % (lr_auc))
# Calculamos las curvas ROC
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# Pintamos las curvas ROC
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='/')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='ROC')
# Etiquetas de los ejes
pyplot.xlabel('Tasa de Falsos Positivos')
pyplot.ylabel('Tasa de Verdaderos Positivos')
pyplot.legend()
pyplot.show()