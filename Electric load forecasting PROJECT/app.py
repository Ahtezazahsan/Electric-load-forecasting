from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

app = Flask(__name__)

df = pd.read_csv('processed_dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

models_dir = 'models'
models = {
    'Linear Regression': joblib.load(os.path.join(models_dir, 'linear_regression_model.joblib')),
    'Random Forest': joblib.load(os.path.join(models_dir, 'random_forest_model.joblib')),
    'XGBoost': joblib.load(os.path.join(models_dir, 'xgboost_model.joblib')),
    'Stacking': joblib.load(os.path.join(models_dir, 'stacking_model.joblib')),
    'ARIMA': joblib.load(os.path.join(models_dir, 'arima_model.joblib'))
}
# kmeans = joblib.load(os.path.join(models_dir, 'kmeans_model.joblib'))
pca = joblib.load(os.path.join(models_dir, 'pca_model.joblib'))
scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))

def ensure_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[columns]

@app.route('/')
def index():
    cities = df['city'].unique().tolist()
    min_date = df['timestamp'].min().strftime('%Y-%m-%d')
    max_date = df['timestamp'].max().strftime('%Y-%m-%d')
    return render_template('index.html', cities=cities, min_date=min_date, max_date=max_date)

@app.route('/get_forecast', methods=['POST'])
def get_forecast():
    data = request.json
    city = data['city']
    start_date = pd.to_datetime(data['start_date'])
    end_date = pd.to_datetime(data['end_date'])
    model_name = data['model']

    mask = (df['city'] == city) & (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    filtered_df = df[mask].copy()
    if filtered_df.empty:
        return jsonify({'error': 'No data available for the selected parameters'})

    features = ['temperature', 'humidity', 'windSpeed', 'hour', 'day_of_week', 'month']
    X = ensure_columns(filtered_df, features)
    if model_name == 'ARIMA':
        arima_data = filtered_df.set_index('timestamp')['demand_mwh']
        preds = models['ARIMA'].forecast(steps=len(arima_data))
        preds = preds.values
    else:
        preds = models[model_name].predict(X)
    return jsonify({
        'dates': filtered_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
        'actual': filtered_df['demand_mwh'].tolist(),
        'predicted': preds.tolist()
    })

@app.route('/get_clusters', methods=['POST'])
def get_clusters():
    data = request.json
    city = data['city']
    start_date = pd.to_datetime(data['start_date'])
    end_date = pd.to_datetime(data['end_date'])
    k = int(data['k'])
    cluster_algo = data.get('cluster_algo', 'kmeans')  # default to kmeans

    mask = (df['city'] == city) & (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    filtered_df = df[mask].copy()
    if filtered_df.empty:
        return jsonify({'error': 'No data available for the selected parameters'})

    cluster_features = ['temperature', 'humidity', 'windSpeed', 'demand_mwh', 'hour', 'day_of_week']
    # fallback for alternate column names
    for alt, orig in [('temperatur', 'temperature'), ('demand_r', 'demand_mwh'), ('day_of_w', 'day_of_week')]:
        if alt in filtered_df.columns:
            cluster_features = [alt if c == orig else c for c in cluster_features]
    X = ensure_columns(filtered_df, cluster_features)
    if X.shape[0] < 2 or np.isnan(X).any().any():
        return jsonify({'error': 'Not enough data or NaNs for clustering'})

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    if cluster_algo == 'kmeans':
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = model.fit_predict(X_scaled)
    elif cluster_algo == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
        clusters = model.fit_predict(X_scaled)
    elif cluster_algo == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=k)
        clusters = model.fit_predict(X_scaled)
    else:
        return jsonify({'error': 'Unknown clustering algorithm'})

    return jsonify({
        'pca1': pca_result[:, 0].tolist(),
        'pca2': pca_result[:, 1].tolist(),
        'clusters': clusters.tolist(),
        'dates': filtered_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
