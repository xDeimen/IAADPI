
import pandas as pd
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import numpy as np

# Load or create the dataset (replace with actual data file)
# Assuming 'industrial_data.csv' is the dataset file
df = pd.read_csv('INDPRO.csv')

# Separate categorical and numerical features for k-prototypes
categorical_columns = ['Category', 'Region']
numerical_columns = ['Production']

# Prepare data for k-prototypes
categorical_data = df[categorical_columns]
numerical_data = df[numerical_columns]
mixed_data = df.to_numpy()

# K-Modes Clustering for purely categorical data
kmodes = KModes(n_clusters=3, init='Huang', n_init=5, verbose=1)
kmodes_labels = kmodes.fit_predict(categorical_data)

# Add labels to the dataset
df['Kmodes_Cluster'] = kmodes_labels

# K-Prototypes Clustering for mixed data types
kprototypes = KPrototypes(n_clusters=3, init='Cao', n_init=5, verbose=1)
kprototypes_labels = kprototypes.fit_predict(mixed_data, categorical=[0, 2])

# Add labels to the dataset
df['Kprototypes_Cluster'] = kprototypes_labels

# Save or display the labeled dataset
print(df)
df.to_csv('labeled_industrial_data.csv', index=False)
