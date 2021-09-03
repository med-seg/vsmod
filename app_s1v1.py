#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 23:32:51 2021

@author: HMA
"""
#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle
import pandas as pd
from scipy.special import inv_boxcox


def computeSubVisFatModel1(age,sex,waist_cm,hip_cm,bmi,weight_kg,ancestry):

    age = int(age) 
    sex = str(sex)
    waist_cm = float (waist_cm)
    hip_cm = float(hip_cm)
    hip_cm = float(hip_cm)
    bmi = float(bmi)
    weight_kg = float(weight_kg)
    ancestry = str(ancestry)

    list_ancestry = ['AFR','AMR','CSA','EAS','EUR','MID']
    list_ancestry.remove(ancestry)
    
    feature_full_list = ['age_when_attended_assessment_centre','Sex','Waist_circumference','Hip_circumference',
                         'Body_mass_index_BMI','weight','Continental_genetic_ancestry']

    test_features=[[age,sex,waist_cm,hip_cm,bmi,weight_kg,ancestry]]
    features = pd.DataFrame(test_features, columns = feature_full_list)
    
    if sex == 'male':
        modelFilename = 'model_1_sub_test.sav'

    else:
        modelFilename = 'model_1_sub_test_female.sav'
        

    data = []
    for i in range(len(list_ancestry)):
      values = [np.nan,sex,np.nan,np.nan,np.nan,np.nan,list_ancestry[i]]
      zipped = zip(feature_full_list, values)
      a_dictionary = dict(zipped)
      print(a_dictionary)
      data.append(a_dictionary)
      
    features = features.append(data, True)
    features_vis = features.copy()
    
    if sex == 'male':
        features.Sex = features.Sex.map({"female":0, "male":1})
    else:
        features= features.drop('Sex', axis = 1)
    
    features = pd.get_dummies(features)
    features = features.dropna()
    features = np.array(features)
    
    model = pickle.load(open(modelFilename, 'rb'))
    sub_fat_value = model.predict(features)
    
    features_vis.Sex = features_vis.Sex.map({"female":0, "male":1})
    features_vis = pd.get_dummies(features_vis)
    features_vis = features_vis.dropna()
    features_vis = np.array(features_vis)
    
    best_lam = 0.42846265290966234
    modelFilename_vis = 'model_1_vis_test.sav'
    model_vis = pickle.load(open(modelFilename_vis, 'rb'))
    vis_fat_value = model_vis.predict(features_vis)
    vis_fat_value0 = inv_boxcox(vis_fat_value,best_lam)

    
    return sub_fat_value, vis_fat_value0


app = Flask(__name__)
# model = pickle.load(open('model_1_sub_test.sav', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    
    
    age = request.values.get('age')
    sex = request.values.getlist('sex')[0]
    waist_cm = request.values.get('waist_cm')
    hip_cm = request.values.get('hip_cm')
    bmi = request.values.get('bmi')
    weight_kg = request.values.get('weight_kg')
    ancestry = request.values.getlist('ancestry')[0]
    
    #For rendering results on HTML GUI
    # string_features = [str(x) for x in request.form.values()]
    # age = int(string_features[0])
    # sex = string_features[1]
    # waist_cm = float(string_features[2])
    # hip_cm = float(string_features[3])
    # bmi = float(string_features[4])
    # weight_kg = float(string_features[5])
    # ancestry = string_features[6]
    
    # ,sex,waist_cm,hip_cm,weight_kg,ancestry
    
    # return render_template(age)

    # int_features = [float(x) for x in request.form.values()]
    prediction, prediction_vis = computeSubVisFatModel1(age,sex,waist_cm,hip_cm,bmi,weight_kg,ancestry)
    
    output = prediction[0]
    output_s = np.round(output/1000,4)

    output_v = prediction_vis[0]
    output_v0 = np.round(output_v/1000,4)

    return render_template('index.html', 
                       sub_prediction_text="Subcutaneous fat: {} \u00D7 10\u00b3 ml".format(output_s),
                       vis_prediction_text="Visceral fat: {} \u00D7 10\u00b3 ml".format(output_v0))

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(debug = False)
    
                               