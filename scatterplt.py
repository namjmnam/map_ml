import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('./rawdata.csv')
scaler = StandardScaler()
df[['X', 'Y']] = scaler.fit_transform(df[['X', 'Y']])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-5, 100)

# Plotting
ax.scatter(df['X'], df['Y'], df['Z'])

# Labels and title
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Scatter Plot')

# Show plot
plt.show()
