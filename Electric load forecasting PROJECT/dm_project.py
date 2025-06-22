# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine Learning and Model Evaluation
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest, StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    silhouette_score, 
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error
)

# Dimensionality Reduction and Clustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage

# Statistical Analysis
from scipy.stats import zscore

# Time Series Analysis
import statsmodels
from statsmodels.tsa.arima.model import ARIMA

# Visualization
import matplotlib.pyplot as plt

# Model Persistence
import joblib
import os

# XGBoost
import xgboost
from xgboost import XGBRegressor

# Print versions of key libraries
print("Library Versions:")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"xgboost: {xgboost.__version__}")
print(f"statsmodels: {statsmodels.__version__}")
print("\nStarting the analysis...\n")

# 1. Load the dataset
print('Loading dataset...')
df = pd.read_csv('merged_dataset1.csv')
print('Columns:', list(df.columns))
print('First 5 rows:')
print(df.head())

# 2. Missing Values: Identify and impute or remove missing entries
print('\nMissing values per column BEFORE imputation:')
print(df.isnull().sum())

# Impute numeric columns with median, categorical with mode
df_num = df.select_dtypes(include=[np.number]).columns
df[df_num] = df[df_num].fillna(df[df_num].median())
df_cat = df.select_dtypes(include='object').columns
for col in df_cat:
    df[col] = df[col].fillna(df[col].mode()[0])

print('\nMissing values per column AFTER imputation:')
print(df.isnull().sum())

# 3. Feature Engineering
print('\nFeature engineering...')
# Convert timestamp to datetime
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    df['season'] = df['month'].apply(get_season)
else:
    print('timestamp column not found!')

# Normalize/scale continuous variables
continuous_cols = [col for col in ['temperature', 'temperatur', 'humidity', 'windSpeed', 'demand_mwh'] if col in df.columns]
scaler = MinMaxScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

print('\nData after feature engineering (first 5 rows):')
print(df.head())

# Save preprocessed data to a new CSV
print('Saving preprocessed data to processed_dataset.csv...')
df.to_csv('processed_dataset.csv', index=False)

# 4. Aggregation: Compute daily and weekly summary statistics
print('\nDaily aggregation:')
if 'timestamp' in df.columns and 'city' in df.columns:
    daily_agg = df.groupby([df['timestamp'].dt.date, 'city']).agg({
        col: 'mean' for col in continuous_cols
    }).reset_index()
    print(daily_agg.head())
    print('\nWeekly aggregation:')
    weekly_agg = df.groupby([df['timestamp'].dt.isocalendar().week, 'city']).agg({
        col: 'mean' for col in continuous_cols
    }).reset_index()
    print(weekly_agg.head())
else:
    print('timestamp or city column not found for aggregation!')

# 5. Anomaly & Error Detection
print('\nAnomaly & Error Detection:')

# Use only continuous columns for anomaly detection
anomaly_cols = continuous_cols

# Z-score method
z_scores = df[anomaly_cols].apply(zscore)
outliers_zscore = (z_scores.abs() > 3).any(axis=1)
print(f'Z-score anomalies found: {outliers_zscore.sum()}')

# IQR method
Q1 = df[anomaly_cols].quantile(0.25)
Q3 = df[anomaly_cols].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((df[anomaly_cols] < (Q1 - 1.5 * IQR)) | (df[anomaly_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
print(f'IQR anomalies found: {outliers_iqr.sum()}')

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers_if = iso_forest.fit_predict(df[anomaly_cols]) == -1
print(f'Isolation Forest anomalies found: {outliers_if.sum()}')

# Combine anomaly flags
anomaly_flags = pd.DataFrame({
    'zscore': outliers_zscore,
    'iqr': outliers_iqr,
    'isoforest': outliers_if
})
df['is_anomaly'] = anomaly_flags.any(axis=1)

print('\nSample flagged anomalies:')
print(df[df['is_anomaly']].head())

# Document anomaly summary
print('\nAnomaly summary:')
anomaly_summary = pd.DataFrame({
    'Method': ['Z-Score', 'IQR', 'Isolation Forest'],
    'Number_of_Anomalies': [outliers_zscore.sum(), outliers_iqr.sum(), outliers_if.sum()]
})
print(anomaly_summary)

# 6. Clustering Task
print('\nClustering Task:')
# Use only non-anomalous data for clustering (optional, or use all)
clustering_df = df[~df['is_anomaly']].copy()

cluster_features = [col for col in ['temperature', 'temperatur', 'humidity', 'windSpeed', 'demand_mwh', 'hour', 'day_of_week'] if col in clustering_df.columns]
X_cluster = clustering_df[cluster_features].values

# 1. Dimensionality Reduction: PCA
print('Performing PCA...')
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_cluster)
plt.figure(figsize=(6,4))
plt.scatter(pca_components[:,0], pca_components[:,1], alpha=0.5)
plt.title('PCA Projection')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# t-SNE
print('Performing t-SNE...')
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_components = tsne.fit_transform(X_cluster)
plt.figure(figsize=(6,4))
plt.scatter(tsne_components[:,0], tsne_components[:,1], alpha=0.5)
plt.title('t-SNE Projection')
plt.xlabel('tSNE-1')
plt.ylabel('tSNE-2')
plt.show()

# 2. K-Means: Elbow method
print('K-Means clustering and elbow method...')
wcss = []
silhouettes = []
k_range = range(2, 8)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    wcss.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_cluster, kmeans.labels_))
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(k_range, wcss, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.subplot(1,2,2)
plt.plot(k_range, silhouettes, 'ro-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.tight_layout()
plt.show()

# Choose optimal k (e.g., 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clustering_df['kmeans_cluster'] = kmeans.fit_predict(X_cluster)
print(f'K-Means cluster counts: {clustering_df.kmeans_cluster.value_counts().to_dict()}')

# Visualize clusters in PCA space
plt.figure(figsize=(6,4))
for i in range(optimal_k):
    plt.scatter(pca_components[clustering_df.kmeans_cluster==i,0], pca_components[clustering_df.kmeans_cluster==i,1], label=f'Cluster {i}', alpha=0.5)
plt.title('K-Means Clusters (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# DBSCAN
print('DBSCAN clustering...')
dbscan = DBSCAN(eps=0.5, min_samples=5)
clustering_df['dbscan_cluster'] = dbscan.fit_predict(X_cluster)
print(f'DBSCAN cluster counts: {clustering_df.dbscan_cluster.value_counts().to_dict()}')
plt.figure(figsize=(6,4))
plt.scatter(pca_components[:,0], pca_components[:,1], c=clustering_df['dbscan_cluster'], cmap='viridis', alpha=0.5)
plt.title('DBSCAN Clusters (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# Hierarchical Clustering
print('Hierarchical clustering and dendrogram...')
# Use a sample for dendrogram to avoid memory errors
max_sample = 1000
if X_cluster.shape[0] > max_sample:
    np.random.seed(42)
    sample_idx = np.random.choice(X_cluster.shape[0], max_sample, replace=False)
    X_hier = X_cluster[sample_idx]
    print(f'Using a random sample of {max_sample} for dendrogram.')
else:
    X_hier = X_cluster
Z = linkage(X_hier, method='ward')
plt.figure(figsize=(10,5))
dendrogram(Z, truncate_mode='lastp', p=12)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# 3. Evaluation: Silhouette score and cluster stability
print('Evaluating cluster stability...')
stability_scores = []
for _ in range(5):
    kmeans = KMeans(n_clusters=optimal_k, random_state=np.random.randint(0,1000))
    labels = kmeans.fit_predict(X_cluster)
    stability_scores.append(silhouette_score(X_cluster, labels))
print(f'Silhouette scores (stability): {stability_scores}')
print(f'Mean: {np.mean(stability_scores):.3f}, Std: {np.std(stability_scores):.3f}')

# 4. Interpretation: Cluster characteristics
print('\nCluster characteristics (K-Means):')
print(clustering_df.groupby('kmeans_cluster')[cluster_features].mean())

# 7. Predictive Modeling
print('\nPredictive Modeling:')

# 1. Problem Formulation: 24-hour ahead forecasting
df = df.sort_values('timestamp')
df['target'] = df['demand_mwh'].shift(-24)
df = df[:-24]  # Remove last 24 rows where target is NaN

# 2. Feature selection
features = [col for col in ['temperature', 'temperatur', 'humidity', 'windSpeed', 'hour', 'day_of_week', 'month'] if col in df.columns]
X = df[features]
y = df['target']

# 3. Train-test split (by date, no shuffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# 4. Baseline: Naive forecast (previous day's same hour)
def naive_forecast(y_true):
    return y_true.shift(24)
naive_preds = naive_forecast(y_test)
naive_mae = mean_absolute_error(y_test[24:], naive_preds[24:])
naive_rmse = np.sqrt(mean_squared_error(y_test[24:], naive_preds[24:]))
naive_mape = mean_absolute_percentage_error(y_test[24:], naive_preds[24:])
print('\nNaive Baseline Results:')
print(f'MAE: {naive_mae:.2f}, RMSE: {naive_rmse:.2f}, MAPE: {naive_mape:.2f}')

# 5. Model selection and training
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}
# Stacking ensemble
stacking = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
        ('lr', LinearRegression())
    ],
    final_estimator=LinearRegression(),
    n_jobs=-1
)
models['Stacking'] = stacking

cv = TimeSeriesSplit(n_splits=3)
results = {}
all_preds = {}

for name, model in models.items():
    print(f'\nTraining {name}...')
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    print(f'Cross-validation RMSE: {np.sqrt(-cv_scores.mean()):.2f} (+/- {np.sqrt(-cv_scores.std()):.2f})')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    all_preds[name] = preds
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f'{name} Results: MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}')

# ARIMA (time series model)
print('\nTraining ARIMA model...')
arima_data = df.set_index('timestamp')['demand_mwh']
arima_model = ARIMA(arima_data, order=(1,1,1))
arima_results = arima_model.fit()
arima_preds = arima_results.forecast(steps=len(y_test))
all_preds['ARIMA'] = arima_preds.values
arima_mae = mean_absolute_error(y_test, arima_preds)
arima_rmse = np.sqrt(mean_squared_error(y_test, arima_preds))
arima_mape = mean_absolute_percentage_error(y_test, arima_preds)
results['ARIMA'] = {'MAE': arima_mae, 'RMSE': arima_rmse, 'MAPE': arima_mape}
print(f'ARIMA Results: MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}, MAPE: {arima_mape:.2f}')

# 6. Evaluation: Plot actual vs. predicted for each model
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual', marker='o')
for name, preds in all_preds.items():
    plt.plot(preds, label=name, marker='x')
plt.title('Model Comparison: Actual vs. Predicted')
plt.xlabel('Hour Index')
plt.ylabel('Demand (MWh)')
plt.legend()
plt.tight_layout()
plt.show()

# Print summary table
print('\nSummary of Model Results:')
print('Model           MAE     RMSE    MAPE')
print(f'Naive      {naive_mae:.2f}   {naive_rmse:.2f}   {naive_mape:.2f}')
for name, res in results.items():
    print(f'{name:14} {res["MAE"]:.2f}   {res["RMSE"]:.2f}   {res["MAPE"]:.2f}')

# Save trained models
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save all trained models
for name, model in models.items():
    joblib.dump(model, f'models/{name.lower().replace(" ", "_")}_model.joblib')

# Save ARIMA model
joblib.dump(arima_results, 'models/arima_model.joblib')

print('\nAll models have been saved to the "models" directory.')

# Save all models including clustering models
print('\nSaving all models...')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save predictive models
for name, model in models.items():
    joblib.dump(model, f'models/{name.lower().replace(" ", "_")}_model.joblib')

# Save ARIMA model
joblib.dump(arima_results, 'models/arima_model.joblib')

# Save clustering models
joblib.dump(kmeans, 'models/kmeans_model.joblib')
joblib.dump(dbscan, 'models/dbscan_model.joblib')

# Save dimensionality reduction models
joblib.dump(pca, 'models/pca_model.joblib')
joblib.dump(tsne, 'models/tsne_model.joblib')

# Save scaler
joblib.dump(scaler, 'models/scaler.joblib')

# Save isolation forest model
joblib.dump(iso_forest, 'models/isolation_forest_model.joblib')

print('\nAll models have been saved to the "models" directory:')
print('1. Predictive Models:')
print('   - Linear Regression')
print('   - Random Forest')
print('   - XGBoost')
print('   - Stacking Ensemble')
print('   - ARIMA')
print('2. Clustering Models:')
print('   - KMeans')
print('   - DBSCAN')
print('3. Dimensionality Reduction:')
print('   - PCA')
print('   - t-SNE')
print('4. Other Models:')
print('   - Scaler')
print('   - Isolation Forest')
