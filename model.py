import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Load the Dataset
df = pd.read_csv('e-commerce2.csv', encoding='ISO-8859-1')

# Data Cleaning
df.dropna(inplace=True)  # Drop rows with missing values
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Feature Engineering
# Calculate TotalPrice per transaction
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Calculate Recency, Frequency, and Monetary (RFM) values per CustomerID
reference_date = df['InvoiceDate'].max()
rfm_data = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                    # Frequency
    'TotalPrice': 'sum'                                        # Monetary
}).reset_index()
rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Data Preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])

# Model Selection - K-Means Clustering
optimal_k = 3  # Set the optimal number of clusters (can be determined via Elbow Method)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Model Evaluation
silhouette_kmeans = silhouette_score(scaled_data, rfm_data['Cluster'])
print(f'Silhouette Score for K-Means: {silhouette_kmeans}')

# Save the Model and Scaler
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully.")
