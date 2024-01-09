from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./rawdata.csv')
scaler = MinMaxScaler()
df[['X', 'Y']] = scaler.fit_transform(df[['X', 'Y']])

descriptive_stats = df.describe()
print(descriptive_stats)

# Assuming 'df' is your DataFrame
X = df[['X', 'Y']]  # Features
y = df['Z']        # Target variable

# Split the data into training + validation and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the training + validation set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

# Building a simple linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Predicting on the validation set
y_val_pred = model.predict(X_val)

# Calculating the Mean Squared Error on the validation set
mse_val = mean_squared_error(y_val, y_val_pred)

print(mse_val)

# Now, once we have tuned our model and are happy with its performance on the validation set,
# we test the model on the test set to get an unbiased evaluation of its performance

# Test the model on the test set
y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)

print(mse_test)

# Generate a grid of x and y values
x_values = np.linspace(df['X'].min(), df['X'].max(), 100)
y_values = np.linspace(df['Y'].min(), df['Y'].max(), 100)
xx, yy = np.meshgrid(x_values, y_values)

# Flatten the grid to pass it through the model
grid = np.vstack([xx.ravel(), yy.ravel()]).T

# Transform the grid using the same scaler used for training
grid_scaled = scaler.transform(grid)

# Predict the z values using the model
zz_pred = model.predict(grid_scaled[:, :2])  # Only take the scaled X and Y

# Reshape the predictions to match the grid shape
zz_pred = zz_pred.reshape(xx.shape)

# Plotting
plt.figure(figsize=(10, 8))
plt.scatter(xx, yy, c=zz_pred, cmap='viridis')
plt.colorbar(label='Predicted Z value')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of predicted Z values on X-Y plane')
plt.show()