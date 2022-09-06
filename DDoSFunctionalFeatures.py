import catboost 
from catboost import CatBoostRegressor 
import shap 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


data_path="combined.csv"
dataset=pd.read_csv(data_path)
array = dataset.values 
X = array[:, :-1] 
Y = array[:, -1]
model = CatBoostRegressor(iterations=500, learning_rate=0.01, random_seed=123) 
model.fit(X, Y, verbose=False, plot=False) 
explainer = shap.TreeExplainer(model) 
shap_values = explainer.shap_values(X) 
vals = np.abs(shap_values).mean(0)
feature_names = dataset.columns

feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                 columns=['col_name','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'],
                              ascending=False, inplace=True)
feature_importance.to_csv("DDoS_Functional_Features.csv")
fig, ax = plt.subplots(figsize=(25,25))
shap.summary_plot(shap_values, X, dataset.columns, plot_type="bar",show=True) 
plt.savefig('featuresImportance.png',bbox_inches = 'tight')
shap.summary_plot(shap_values, X, dataset.columns) 
plt.savefig('featuresImpact.png',bbox_inches = 'tight')

