#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:04:36 2019

@author: user
"""


from pynlpl.formats import folia
import glob
import os
import json
import pandas as pd


all_path = os.popen('find . -name "adjudication"').readlines()
for idx, elem in enumerate(all_path):
    all_path[idx] = elem[:-1]


for idx, elem in enumerate(all_path):
    all_path[idx] = "/home/user/Desktop/corpusannotation/corpus/Token" + elem[1:]


adjudicated_all = dict()


yes_file_list = []
no_file_list = []
for folder_path in all_path:
    os.chdir(folder_path)
    print(folder_path)
    adjudicated = dict()
    file_list = []
    for elem in glob.glob("*.xml"):
        doc = folia.Document(file=elem)
        data = dict(doc.metadata.data)
        try:
            print(data['Violent'])
            if(data['Violent'] == 'Yes'):
                yes_file_list.append([elem, "Yes"])
                file_list.append([elem, "Yes"])
            if(data['Violent'] == 'No'):
                no_file_list.append([elem, "No"])
                file_list.append([elem, "No"])
        except:
            print("There is No Violent")
        adjudicated['data'] = file_list
    adjudicated_all[folder_path] = adjudicated
    


def compare(first, second):
    os.chdir(first)
    first_annotations = []
    for elem in glob.glob("*.xml"):
        doc = folia.Document(file=elem)
        try:
            first_annotations.append(doc.metadata.data['Violent'])
        except:
            first_annotations.append("Annotation Empty")
    os.chdir(second)
    second_annotations = []
    for elem in glob.glob("*.xml"):
        doc = folia.Document(file=elem)
        try:
            second_annotations.append(doc.metadata.data['Violent'])
        except:
            second_annotations.append("Annotation Empty")

    first_annotator_final = []
    second_annotator_final = []
    first_not_labeled = []
    second_not_labeled = []
    for idx, elem in enumerate(first_annotations):
        if (elem == "Yes" or elem == "No") and (second_annotations[idx] == "Yes" or second_annotations[idx] == "No"):
            first_annotator_final.append(elem)
            second_annotator_final.append(second_annotations[idx])
        else:
            first_not_labeled.append(elem)
            second_not_labeled.append(second_annotations[idx])
            
            
    
    return [[first_annotator_final, second_annotator_final], [first_not_labeled, second_not_labeled]]
        #print(doc.metadata.data['Violent'])


def get_annotations(first, second, adjudication_path):
    os.chdir(first)
    filename = []
    for elem in glob.glob("*.xml"):
        filename.append(elem)
        
    filetext = []
    for elem in glob.glob("*.xml"):
        doc = folia.Document(file=elem)
        filetext.append(doc.text())
    
    first_annotations = []
    for elem in glob.glob("*.xml"):
        doc = folia.Document(file=elem)
        try:
            first_annotations.append(doc.metadata.data['Violent'])
        except:
            first_annotations.append("Annotation Empty")
    os.chdir(second)
    second_annotations = []
    for elem in glob.glob("*.xml"):
        doc = folia.Document(file=elem)
        try:
            second_annotations.append(doc.metadata.data['Violent'])
        except:
            second_annotations.append("Annotation Empty")
    
    adjudication = []
    os.chdir(adjudication_path)
    for elem in filename:
        doc = folia.Document(file=elem)
        try:
            adjudication.append(doc.metadata.data['Violent'])
        except:
            adjudication.append("Adjudication Empty")
            
    zip_list = list(zip(filename, filetext, first_annotations, second_annotations, adjudication))
    df = pd.DataFrame(zip_list, columns = ["Filename", "File Text","First", "Second", "Adjudication"])
    return df
    



from sklearn.metrics import cohen_kappa_score

#Eylem Ezgi Indian Express
eylem_indianexpress = "/home/user/Desktop/corpusannotation/corpus/Token/ezgi-eylem_20180711_indianexpress/Eylem"
ezgi_indianexpress = "/home/user/Desktop/corpusannotation/corpus/Token/ezgi-eylem_20180711_indianexpress/Ezgi"
eylem_ezgi_indianexpress_adjudication = "/home/user/Desktop/corpusannotation/corpus/Token/ezgi-eylem_20180711_indianexpress/adjudication"

#Eylem Ezgi Thehindu
eylem_thehindu = "/home/user/Desktop/corpusannotation/corpus/Token/ezgi-eylem_20180801_thehindu/Eylem"
ezgi_thehindu = "/home/user/Desktop/corpusannotation/corpus/Token/ezgi-eylem_20180801_thehindu/Ezgi"
eylem_ezgi_thehindu_adjudication = "/home/user/Desktop/corpusannotation/corpus/Token/ezgi-eylem_20180801_thehindu/adjudication"

#Pelin Selim IndianExpress
pelin_indianexpress = "/home/user/Desktop/corpusannotation/corpus/Token/pelin-selim_20180711_indianexpress/Pelin"
selim_indianexpress = "/home/user/Desktop/corpusannotation/corpus/Token/pelin-selim_20180711_indianexpress/selim"
pelin_selim_indianexpress_adjudication = "/home/user/Desktop/corpusannotation/corpus/Token/pelin-selim_20180711_indianexpress/adjudication"

#Sercan Gizem IndianExpress
sercan_indianexpress = "/home/user/Desktop/corpusannotation/corpus/Token/sercan-gizem_20180711_indianexpress/Sercan"
gizem_indianexpress = "/home/user/Desktop/corpusannotation/corpus/Token/sercan-gizem_20180711_indianexpress/Gizem"
sercan_gizem_indianexpress_adjucation = "/home/user/Desktop/corpusannotation/corpus/Token/sercan-gizem_20180711_indianexpress/adjudication"

#Sercan Gizem Newindianexpress
sercan_newindian = "/home/user/Desktop/corpusannotation/corpus/Token/sercan-gizem_20180919_newindianexpress/Sercan"
gizem_newindian = "/home/user/Desktop/corpusannotation/corpus/Token/sercan-gizem_20180919_newindianexpress/Gizem"
sercan_gizem_newindian_adjudication = "/home/user/Desktop/corpusannotation/corpus/Token/sercan-gizem_20180919_newindianexpress/adjudication"

a = compare(eylem_indianexpress, ezgi_indianexpress, eyle)
b = compare(eylem_thehindu, ezgi_thehindu)
c = compare(pelin_indianexpress, selim_indianexpress)
d = compare(sercan_indianexpress, gizem_indianexpress)
e = compare(sercan_newindian, gizem_newindian)

print(cohen_kappa_score(a[0][0], a[0][1]))
print(cohen_kappa_score(b[0][0], b[0][1]))
print(cohen_kappa_score(c[0][0], c[0][1]))
print(cohen_kappa_score(d[0][0], d[0][1]))
print(cohen_kappa_score(e[0][0], e[0][1]))



first_an = sum([a[0][0], b[0][0], c[0][0], d[0][0], e[0][0]], [])
second_an = sum([a[0][1], b[0][1], c[0][1], d[0][1], e[0][1]], [])

print(cohen_kappa_score(first_an, second_an))


df_a = get_annotations(eylem_indianexpress, ezgi_indianexpress, eylem_ezgi_indianexpress_adjudication )
df_b = get_annotations(eylem_thehindu, ezgi_thehindu, eylem_ezgi_thehindu_adjudication)
df_c = get_annotations(pelin_indianexpress, selim_indianexpress, pelin_selim_indianexpress_adjudication)
df_d = get_annotations(sercan_indianexpress, gizem_indianexpress, sercan_gizem_indianexpress_adjucation)
df_e = get_annotations(sercan_newindian, gizem_newindian, sercan_gizem_newindian_adjudication)


df = df_a.append([df_b, df_c, df_d, df_e])


df.to_excel("annotation.xlsx")