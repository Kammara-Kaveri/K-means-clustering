##TASK -8
# Task 8: K-Means Clustering

## Objective
This project demonstrates how to apply **K-Means Clustering** for customer segmentation using unsupervised learning techniques.

## Tools Used
- Python
- Pandas
- Scikit-learn
- Matplotlib

##Libraries installed
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


## Steps Followed
1. **Loaded the Dataset** – Loaded a customer-related dataset suitable for segmentation.
2. **Data Preprocessing** – Cleaned and scaled the data using `StandardScaler`.
3. **Elbow Method** – Used the elbow curve to find the optimal number of clusters (K).
4. **Model Training** – Applied `KMeans` clustering to the data.
5. **Visualization** – Plotted clusters with color-coding to visualize the grouping.
6. **Evaluation** – Calculated **Silhouette Score** to evaluate cluster quality.



```bash
pip install -r requirements.txt

##
Then run the Python script:
python k_cluster.py

## Steps Followed

1. Loaded and explored the dataset.
2. Preprocessed the data (including scaling).
3. Applied K-Means clustering.
4. Used the Elbow Method to find the optimal number of clusters.
5. Visualized the clusters.
6. Evaluated the model using the Silhouette Score.

##Output
Elbow curve plot to determine K.
Cluster visualization (2D plot).
Printed silhouette score.

##Learnings
How K-Means groups data points based on similarity.
How to evaluate clusters using the Silhouette Score.
Use of the Elbow Method for finding optimal cluster count.

##Dataset
Dataset used: Mall_Customers.csv

##Author:
GitHub:@kammarakaveri

