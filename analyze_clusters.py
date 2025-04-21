import pandas as pd
import joblib

# Load the model and scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Analyze cluster characteristics
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_summary = pd.DataFrame(cluster_centers, columns=['Recency', 'Frequency', 'Monetary'])
cluster_summary['Cluster'] = range(len(cluster_summary))  # Add cluster number

print("Cluster Summary:")
print(cluster_summary)
