import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Create the dataset
data={
    'Size':[140,150,170,200,250],
    'Price':[100000,110000,130000,150000,200000]
}

df=pd.DataFrame(data)
X=data['Size']
y=data['Price']

# Visualize the data
plt.scatter(X,y,color="red",marker="o")
plt.title("House Prediction")
plt.xlabel("size(M)")
plt.ylabel("price($)")
plt.show()

plt.legend()
plt.show()

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Make predictions and visualize
predictions = model.predict(X)
plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Linear Regression Prediction')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(y, predictions)
print("MSE:", mse)
