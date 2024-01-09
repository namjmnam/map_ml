from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./rawdata.csv')
# scaler = MinMaxScaler()
scaler = StandardScaler()
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

# Initialize the KNN regressor
# You can change the number of neighbors (n_neighbors)
model = KNeighborsRegressor(n_neighbors=4)

# Train the model
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

# Sample data for demonstration, replace these with your actual data
# df['X'], df['Y'] and the model predictions
zz = np.sin(xx) + np.cos(yy)  # Replace this with your model's zz_pred

# Creating a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(xx, yy, zz, c=zz, cmap='viridis')

# Adding labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter plot of predicted Z values on X-Y plane')

# Adding a color bar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Predicted Z value')

plt.show()

# Generating sample data for demonstration
# This sample data will act as a stand-in for the actual data in 'rawdata.csv'
raw_df = pd.read_csv('./rawdata.csv')
raw_df[['X', 'Y']] = scaler.fit_transform(raw_df[['X', 'Y']])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the first scatter plot (from the grid data)
scatter1 = ax.scatter(xx, yy, zz, c=zz, cmap='viridis', label='Predicted Data')

# Plotting the second scatter plot (from the sample raw data)
scatter2 = ax.scatter(raw_df['X'], raw_df['Y'], raw_df['Z'], c='red', label='Raw Data')

# Labels and title
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Combined 3D Scatter Plot')

# Setting the z-axis limit
ax.set_zlim(-5, 50)

# Adding a legend
ax.legend()

# Show plot
plt.show()
