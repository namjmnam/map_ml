import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

def process_csv(file_path, n):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Normalize x and y columns
    scaler = MinMaxScaler()
    df[['X', 'Y']] = scaler.fit_transform(df[['X', 'Y']])

    # Pick n random entries from the table
    sampled_df = df.sample(n=n)

    return sampled_df

result = process_csv('./rawdata.csv', 30)
print(result)

# Creating a scatter plot
plt.scatter(result['X'], result['Y'])

# Adding titles and labels
plt.title('Scatter Plot of Normalized X and Y')
plt.xlabel('Normalized X')
plt.ylabel('Normalized Y')

# Displaying the plot
plt.show()