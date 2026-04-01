import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('dataset/data.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Correlation analysis
print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# Visualizations
plt.figure(figsize=(12, 6))

# Heatmap of correlations
plt.subplot(1, 2, 1)
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')

# Missing values
plt.subplot(1, 2, 2)
df.isnull().sum().plot(kind='bar')
plt.title('Missing Values by Column')
plt.tight_layout()
plt.show()

# Distribution plots for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols].hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()