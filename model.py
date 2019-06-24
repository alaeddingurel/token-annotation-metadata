#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:48:08 2019

@author: user
"""

from pynlpl.formats import folia
import glob
import os
import json
import pandas as pd

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score




all_path = os.popen('find . -name "adjudication"').readlines()
for idx, elem in enumerate(all_path):
    all_path[idx] = elem[:-1]


for idx, elem in enumerate(all_path):
    all_path[idx] = "/home/user/Desktop/corpusannotation/corpus/Token" + elem[1:]


adjudicated_all = dict()


text_file = []
text_data = []
violent_list = []

for folder_path in all_path:
    os.chdir(folder_path)
    for elem in glob.glob("*.xml"):
        doc = folia.Document(file=elem)
        try:
            print(doc.metadata.data['Violent'])
            text_data.append(doc.text())
            violent_list.append(doc.metadata.data['Violent'])
            text_file.append(elem)
        except:
            print("File will not be trained")
            
zip_list = list(zip(text_file, text_data, violent_list))
new_df = pd.DataFrame(zip_list, columns = ["Filename", "Text", "etype"])

new_df['Text'].replace(regex=True,inplace=True,to_replace=r'url : (.)*',value=r'')


new_df.loc[new_df.etype == "Yes", "etype"] = 1
new_df.loc[new_df.etype == "No", "etype"] = 0

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
kfold = KFold(10, True, 1)

from sklearn.metrics import f1_score

def dummy_fun(doc):
    return doc

df = pd.DataFrame()


for train, test in kfold.split(new_df):
    X_train = new_df.iloc[train]['Text']
    y_train = new_df.iloc[train]['etype']
    X_test = new_df.iloc[test]['Text']
    y_test = new_df.iloc[test]['etype']
    
    X_test_docfile = new_df.iloc[test]['Filename']
    
    #print(X_test)
 
    
    tfidf_vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), tokenizer=dummy_fun, preprocessor=dummy_fun, analyzer='word', token_pattern=r'\w{1,}', max_features=20000)
    train_tfidf = tfidf_vect.fit_transform(X_train)
    test_tfidf = tfidf_vect.transform(X_test)
    
    gsc = GridSearchCV(
        estimator=svm.LinearSVC(),
        param_grid={
            'C': [0.1, 1, 10, 100, 1000]
        },
        cv=10, scoring="accuracy", verbose=1, n_jobs=-1)
    grid_result = gsc.fit(train_tfidf, list(y_train))

    best_params = grid_result.best_params_
    best_svr = svm.LinearSVC(C=best_params["C"])
    best_svr.fit(train_tfidf, list(y_train))
    svm_predicted = best_svr.predict(test_tfidf)
        
    df = pd.DataFrame()
    df['doc'] = X_test_docfile
    df['X_test'] = X_test
    df['y_test'] = y_test
    df['prediction'] = svm_predicted
    precision = recall_score(list(y_test), svm_predicted)
    recall = recall_score(list(y_test), svm_predicted)
    f1 = f1_score(list(y_test), svm_predicted)
    acc = accuracy_score(list(y_test), svm_predicted)
    
    df.append(pd.Series([precision, recall, f1, acc], index=df.columns), ignore_index=True)

    df.to_excel(X_test.iloc[0][0:20] + ".xlsx")
    
    
        