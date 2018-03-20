#import 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import csv



# read data 
data = pd.read_csv('train_modified2.csv')


#Modify Dataframe 


data["pickup_date_time"] = pd.to_datetime(data["pickup_date_time"])
data = data.copy()
data['hours'] = data['pickup_date_time'].dt.hour
data['day']= data['pickup_date_time'].dt.day

data['month']= data['pickup_date_time'].dt.month

data['temperature']= -1


# ====================== Train and Test



columns_tain = ['ct2010','hours','day','weekend']

columns_target = ['nb_pickup']

X = data.loc[:,columns_tain].values

Y = data.loc[:,columns_target].values



x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)



# Model


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state = 0)

model.fit(x_train, y_train)



y_pred=np.int64(model.predict(x_test))

out = pd.DataFrame(model.predict(x_test))




# submit



pd3 = pd.read_csv('submission_sample.csv')

pd3 = pd3.copy()



pd3['ct2010'] = np.int64(str(pd3['key']).split("/")[0].strip().replace(" ", ""))

pd3['pickup_date_time'] =str(pd3['key']).split("/")[1].strip()

pd3['pickup_date_time'] = pd.to_datetime(data["pickup_date_time"])

pd3['hours'] = pd3['pickup_date_time'].dt.hour

pd3['day']= pd3['pickup_date_time'].dt.day

pd3['month']= pd3['pickup_date_time'].dt.month

pd3['temperature']= 0
pd3['weekend']= data['weekend']
pd3['degre']= -1

pd3.to_csv('sub_modified.csv',encoding='utf-8')

test = pd.read_csv('sub_modified.csv')

x_new = test.loc[:,columns_tain].values



new_pred_class = model.predict(x_new)



pd.DataFrame({'key':pd3.key,'nb_pickup':np.int64(new_pred_class)}).set_index('key').to_csv('submission.csv')









