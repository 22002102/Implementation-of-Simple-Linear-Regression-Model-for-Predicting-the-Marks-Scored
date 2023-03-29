# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3.  Implement training set and test set of the dataframe
4.  Plot the required graph both for test data and training data.


## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SANJAY S
RegisterNumber:212222230132

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile
df.head()

df.tail()

x=df.iloc[:,:-1].values
print(x)

y=df.iloc[:,1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)


```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


![1](https://user-images.githubusercontent.com/119091638/228565656-91ba5a8e-5dca-42f8-8538-483d53f584f6.png)

![2](https://user-images.githubusercontent.com/119091638/228565705-da988b07-9535-42f9-a268-ebd87e7cd51f.png)

![3](https://user-images.githubusercontent.com/119091638/228565766-ae343b23-bbc5-4a68-99e5-ef324e1f0fe8.png)

![4](https://user-images.githubusercontent.com/119091638/228565867-8182e04f-7044-473c-9eba-ea58b854dfc5.png)

![5](https://user-images.githubusercontent.com/119091638/228565921-cb5ae454-1888-49ce-a508-46ac6f12dc9d.png)

![6](https://user-images.githubusercontent.com/119091638/228565965-8a40858d-c210-4a34-b3ed-beda7d60bb6c.png)

![7](https://user-images.githubusercontent.com/119091638/228566017-8b48e63a-c2f4-4975-be12-f9effbb2f1b7.png)

![8](https://user-images.githubusercontent.com/119091638/228566048-354bccdd-f35d-4652-819c-b6b9ef7552c2.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
