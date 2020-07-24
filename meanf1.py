from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import pandas as pd


def meanf1(list_of_algorithms,x,y):
    score_total=0
    i=0
    
    for algorithm in list_of_algorithms:
        
        scores = cross_val_score(algorithm, x, y, cv=4, scoring='f1_macro')
        score_total=scores.mean()+score_total
        i=i+1
        
    print(score_total/i)
    

