from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from django.core.files.storage import FileSystemStorage


#model imports
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score,confusion_matrix,f1_score,recall_score

gnb = GaussianNB()
svm = SVC()
dt = tree.DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators = 25)
ada = AdaBoostClassifier()
lr=LogisticRegression(random_state=0)



# Create your views here.

def analysefun(datasetSelected,datasetFile,datasetOuputCol,classifier,kval):
    res=[]
    global df,x,y,X_train, X_test,y_train,y_test
    cm=[]
    accuracy=-1
    precisons=-1
    f1score=-1
    recall=-1
    dataset=datasetSelected
    if  dataset=="diabetes":
        df = pd.read_csv("csvfiles/diabetes.csv")
        y = df['Outcome']
        x = df.drop("Outcome",axis = 1)
        X_train, X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
    elif dataset=="heartStroke":
        df = pd.read_csv("csvfiles/strokes.csv")
        y = df['stroke']
        x = df.drop("stroke",axis = 1)
        X_train, X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
    elif dataset=="customDataset":
        fs=FileSystemStorage()
        filePathName=fs.save(datasetFile.name,datasetFile)
        filePathName=fs.url(filePathName)
        filePath='.'+filePathName
        df = pd.read_csv(filePath)
        y = df[datasetOuputCol]
        x = df.drop(datasetOuputCol,axis = 1)
        X_train, X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)

    if classifier=="svm":
        svm.fit(X_train,y_train)
        y_pred_svm=svm.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred_svm)
        precisons=precision_score(y_test,y_pred_svm,average='weighted')
        cm=confusion_matrix(y_test,y_pred_svm)
        f1score=f1_score(y_test,y_pred_svm,average='weighted')
        recall=recall_score(y_test,y_pred_svm,average='weighted')
        res.append({'classifier':classifier,'accuracy':accuracy*100,'precisons':precisons,'f1score':f1score,'recall':recall,'cm':cm.tolist()})

    elif classifier=="decisionTree":
        dt.fit(X_train,y_train)
        y_pred_dt=dt.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred_dt)
        precisons=precision_score(y_test,y_pred_dt,average='weighted')
        cm=confusion_matrix(y_test,y_pred_dt)
        f1score=f1_score(y_test,y_pred_dt,average='weighted')
        recall=recall_score(y_test,y_pred_dt,average='weighted')
        res.append({'classifier':classifier,'accuracy':accuracy*100,'precisons':precisons,'f1score':f1score,'recall':recall,'cm':cm.tolist()})
    elif classifier=="randomForest":
        rf.fit(X_train,y_train)
        y_pred_rf=rf.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred_rf)
        precisons=precision_score(y_test,y_pred_rf,average='weighted')
        cm=confusion_matrix(y_test,y_pred_rf)
        f1score=f1_score(y_test,y_pred_rf,average='weighted')
        recall=recall_score(y_test,y_pred_rf,average='weighted')
        res.append({'classifier':classifier,'accuracy':accuracy*100,'precisons':precisons,'f1score':f1score,'recall':recall,'cm':cm.tolist()})
    elif classifier=="naiveBayes":
        gnb.fit(X_train,y_train)
        y_pred_gnb=gnb.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred_gnb)
        precisons=precision_score(y_test,y_pred_gnb,average='weighted')
        cm=confusion_matrix(y_test,y_pred_gnb)
        f1score=f1_score(y_test,y_pred_gnb,average='weighted')
        recall=recall_score(y_test,y_pred_gnb,average='weighted')
        res.append({'classifier':classifier,'accuracy':accuracy*100,'precisons':precisons,'f1score':f1score,'recall':recall,'cm':cm.tolist()})
    elif classifier=="knn":
        knn = KNeighborsClassifier(n_neighbors=int(kval))
        knn.fit(X_train,y_train)
        y_pred_knn=knn.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred_knn)
        precisons=precision_score(y_test,y_pred_knn,average='weighted')
        cm=confusion_matrix(y_test,y_pred_knn)
        f1score=f1_score(y_test,y_pred_knn,average='weighted')
        recall=recall_score(y_test,y_pred_knn,average='weighted')
        res.append({'classifier':classifier,'accuracy':accuracy*100,'precisons':precisons,'f1score':f1score,'recall':recall,'cm':cm.tolist()})
    elif classifier=="adaBoost":
        ada.fit(X_train,y_train)
        y_pred_ada=ada.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred_ada)
        precisons=precision_score(y_test,y_pred_ada,average='weighted')
        cm=confusion_matrix(y_test,y_pred_ada)
        f1score=f1_score(y_test,y_pred_ada,average='weighted')
        recall=recall_score(y_test,y_pred_ada,average='weighted')
        res.append({'classifier':classifier,'accuracy':accuracy*100,'precisons':precisons,'f1score':f1score,'recall':recall,'cm':cm.tolist()})
    elif classifier=="logisticRegression":
        lr.fit(X_train,y_train)
        y_pred_lr=lr.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred_lr)
        precisons=precision_score(y_test,y_pred_lr,average='weighted')
        cm=confusion_matrix(y_test,y_pred_lr)
        f1score=f1_score(y_test,y_pred_lr,average='weighted')
        recall=recall_score(y_test,y_pred_lr,average='weighted')
        res.append({'classifier':classifier,'accuracy':accuracy*100,'precisons':precisons,'f1score':f1score,'recall':recall,'cm':cm.tolist()})
    return res


def analysePage(request):
    result =[]
    dataset_data=[]
    global datasetSelected,datasetFile,datasetOuputCol,classifierList
    datasetFile=""
    datasetOuputCol=""
    #taking data from frontend
    datasetSelected = request.POST['datasetSel'] 
    if datasetSelected=='customDataset':
        datasetFile=request.FILES['datasetFile']
        #print(datasetFile)
        datasetOuputCol=request.POST['datasetOuputCol']
    classifierList=json.loads(request.POST['classifierList'])

    
    dataset_data.append({'datasetSelected':datasetSelected})
    result.append(dataset_data)
   
    #print((classifierList))
    for i in classifierList:
        classifier=i['select']
        kval=i['number']
        temp=analysefun(datasetSelected,datasetFile,datasetOuputCol,classifier,kval)
        result.append(temp)
    return JsonResponse(result,safe=False)

    #calling the function to analyse the dataset
    
    #print(datasetFile)
    #print(datasetSelected)
    #print(datasetOuputCol)
    #print(classifierList)
    return JsonResponse(result, safe=False)

def test(get,post):
    return JsonResponse({'test':'test'}, safe=False)


