import os
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response

# Create your views here.

import numpy as np 
import pandas as pd 
import sklearn 
import matplotlib.pyplot as pt
from sklearn import preprocessing
loc_data = "{base_path}/data/heart.xlsx".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))
dataset=pd.read_excel(loc_data)

X=np.asarray(dataset)
a,b=X.shape
r=b-1
# r=0
atribut=X[:,0:r]
target=X[:,r]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(atribut, target, test_size=0.4, random_state=42)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

@api_view(['GET'])
def add(request):
    if request.method == 'GET':
        model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                               random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                               class_weight=None, ccp_alpha=0.0)
        model.fit(X_train,y_train) 
        age = request.GET.get('age')
        sex = request.GET.get('sex')
        cpt = request.GET.get('cpt')
        rbp = request.GET.get('rbp')
        chl = request.GET.get('chl')
        fbs = request.GET.get('fbs')
        recg = request.GET.get('recg')
        mhr = request.GET.get('mhr')
        ea = request.GET.get('ea')
        op = request.GET.get('op')
        sts = request.GET.get('sts')
        input = [[age,sex,cpt,rbp,chl,fbs,recg,mhr,ea,op,sts]]
        ypred=model.predict(input)
        ypred1 = ypred.shape 
        a = 1
        if ypred == a :
            data={
                'pred': f"{1}"
            }
            return JsonResponse(data)
        else :
            data={
                'pred': f"{0}"
            }
            return JsonResponse(data)

        # data={
        #         'pred': f"{ypred}"
        #     }
        # return JsonResponse(data)

        

# model = KNeighborsRegressor(n_neighbors=9, metric='euclidean') 
# model = MLPClassifier(hidden_layer_sizes=(100), activation='logistic',solver='adam')
