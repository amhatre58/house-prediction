import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
# 1. Load the Dataset
data=pd.read_csv('kc_house_data.csv')
print(data.head())
# 2. Data Cleaning
print("Missing values per column:\n",data.isnull().sum())
data['sqft_above']=data['sqft_above'].fillna(data['sqft_above'].median())
data['date'] = pd.to_datetime(data['date'])
print("Data cleaning complete! Cleaned dataset shape:",data.shape)
# 3. Getting Age of the House
data['year_sold']=data['date'].dt.year
data['effective_year_built'] = np.where(data['yr_renovated']==0, data['yr_built'],data['yr_renovated'])
data['house_age']=data['year_sold']-data['effective_year_built']
data.loc[data['house_age'] < 0, 'house_age'] = 0
# 4. Feature Selection
project_features=[
    'bedrooms',
    'house_age',
    'zipcode',
    'sqft_living']
X=data[project_features]
y=data['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# 5. Decision Tree
dt_model=DecisionTreeClassifier(criterion='gini',max_depth=5,random_state=42)
dt_model.fit(X_train,y_train)
y_pred_dt=dt_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print("Comparing Meterics:\nDecision Tree meterics:")
print('Mean Squared Error:',mse_dt)
print('Root Mean Squared Error:',rmse_dt)
print('R-squared:',r2_dt,'\n')
# 6. Random Forest Tree
rf_model=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42)
rf_model.fit(X_train,y_train)
y_pred_rf=rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("Random Forest Meterics:")
print('Mean Squared Error:',mse_rf)
print('Root Mean Squared Error:',rmse_rf)
print('R-squared:',r2_rf,'\n')
# 7. Feature Importance
importances=rf_model.feature_importances_
feature_names=X.columns
forest_importances=pd.Series(importances,index=feature_names).sort_values(ascending=False)
# 8. Visualization
forest_importances.plot(kind='bar',color="skyblue")
plt.title('Feature Importances for House Price Prediction')
plt.ylabel('Importance Weight Score')
plt.xlabel('Features')
plt.tight_layout()
plt.show()
