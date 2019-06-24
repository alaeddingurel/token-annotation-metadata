#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:41:14 2019

@author: user
"""
from pynlpl.formats import folia
import glob
import os
import json
import pandas as pd

def gs_svm(train, y_train, scoring="recall", f1_param="macro"):
    model_dict = dict()
    gsc = GridSearchCV(
        estimator=svm.LinearSVC(),
        param_grid={
            'C': [0.1, 1, 10, 100, 1000]
        },
        cv=10, scoring=scoring, verbose=1, n_jobs=-1)
    grid_result = gsc.fit(train, list(y_train))

    best_params = grid_result.best_params_
    best_svr = svm.LinearSVC(C=best_params["C"])
    best_svr.fit(train, list(y_train))
    svm_predicted = best_svr.predict(test_tfidf)
    
    model_dict['model'] = best_svr
    model_dict['prediction'] = svm_predicted
    
    
    print("SVM F1 Score : ")
    print(f1_score(list(y_test), svm_predicted, average=f1_param))
    print("SVM Accuracy Score : ")
    print(accuracy_score(list(y_test), svm_predicted))
    print("SVM Recall : ")
    print(recall_score(list(y_test), svm_predicted))
    print("SVM Precision : ")
    print(precision_score(list(y_test), svm_predicted))


all_path = os.popen('find . -name "adjudication"').readlines()
for idx, elem in enumerate(all_path):
    all_path[idx] = elem[:-1]


for idx, elem in enumerate(all_path):
    all_path[idx] = "/home/user/Desktop/corpusannotation/corpus/Token" + elem[1:]


adjudicated_all = dict()



from sklearn.model_selection import cross_val_score


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





from sklearn.metrics import precision_recall_curve
print(precision_recall_curve(list(y_test), predicted))

new_df.loc[new_df.etype == "Yes", "etype"] = 1
new_df.loc[new_df.etype == "No", "etype"] = 0


import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

new_df.Text = new_df.Text.apply(lemmatize_text)

df_X = new_df['Text']
df_Y = new_df['etype']
df_fina = pd.concat([df_X, df_Y], axis=1, sort=False)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state=42)


def dummy_fun(doc):
    return doc



from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), tokenizer=dummy_fun, preprocessor=dummy_fun, analyzer='word', token_pattern=r'\w{1,}', max_features=20000)
train_tfidf = tfidf_vect.fit_transform(X_train)
test_tfidf = tfidf_vect.transform(X_test)

from sklearn import svm
clf = svm.LinearSVC(penalty='l2', dual=False)
clf.fit(train_tfidf, list(y_train))
predicted = clf.predict(test_tfidf)


scores = cross_val_score(clf, train_tfidf, list(y_train), cv=10, scoring="f1_macro")
scores.mean()

from sklearn.metrics import f1_score
print("F1 Score : ")
print(f1_score(list(y_test), predicted))
from sklearn.metrics import accuracy_score
print("Accuracy Score : ")
print(accuracy_score(list(y_test), predicted))



from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print(precision_score(list(y_test), predicted))
print(recall_score(list(y_test), predicted))


analysis_df = pd.DataFrame()

analysis_df['text'] = X_test
analysis_df['etype'] = y_test
analysis_df['predicted'] = predicted

"""
oldStory/36100
http__archive.indianexpress.com_oldStory_36100_.folia.xml
http__archive.indianexpress.com_news_nsui-demands-cbi-probe-into-family-s-suicide_511751_.folia.xml
"""

y_train = y_train.astype('int')
from sklearn.model_selection import GridSearchCV
svr = gs_svm(train_tfidf, y_train)