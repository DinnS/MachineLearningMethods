import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# read .csv into a Data frame
house_data = pd.read_csv("house_price.csv")
size = house_data['sqft_living']
price = house_data['price']

# Convert to array

x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)

# Use Linear Regression and training
model = LinearRegression()
model.fit(x,y)

# MSE and R value
regression_model_mse = mean_squared_error(x,y)
print('MSE : ',math.sqrt(regression_model_mse))
print('R squared value : ',model.score(x,y))

# b0
print(model.intercept_[0])
# b1
print(model.coef_[0])

# Visualize data
plt.scatter(x,y,color = 'green')
plt.plot(x,model.predict(x),color = 'black')
plt.title('Linear Regression')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()

# Predict data
print('Prediction by the model : ',model.predict([[1600]]))