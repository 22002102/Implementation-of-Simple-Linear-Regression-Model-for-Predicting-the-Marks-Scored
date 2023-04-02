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



![L1](https://user-images.githubusercontent.com/119091638/229136565-960a410a-8b6e-4b14-bd0f-4d11b39ff6c2.png)

![L2](https://user-images.githubusercontent.com/119091638/229136601-8e28b11b-c3db-4e50-9d99-9ed0feba0b75.png)

![L3](https://user-images.githubusercontent.com/119091638/229136633-bba1280c-975c-4e1c-a827-fb9198e0f575.png)

![L4](https://user-images.githubusercontent.com/119091638/229136685-20985525-abec-4e8b-a65d-b0ee73484e22.png)

![m1](https://user-images.githubusercontent.com/119091638/229324592-72e25c53-2188-4dae-a5e2-b0f1a68d38d0.png)

![m2](https://user-images.githubusercontent.com/119091638/229324601-fad6b730-9f97-4fa4-99a8-21c21dc8c114.png)

![L6](https://user-images.githubusercontent.com/119091638/229136738-7eb3211d-2010-4b87-b579-48b7eafce423.png)

![L7](https://user-images.githubusercontent.com/119091638/229136756-462f7f04-72f9-48ec-a1c5-5c1535306beb.png)

![L8](https://user-images.githubusercontent.com/119091638/229136787-0e7290fb-48c3-4319-a2c8-d6b89f29165e.png)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
