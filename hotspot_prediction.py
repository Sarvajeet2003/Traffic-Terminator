import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import folium
from folium.plugins import HeatMap
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
print("Loading data...")
driver_data = pd.read_csv('driver_data.csv')
ride_data = pd.read_csv('ride_data.csv')

# Display basic information about the datasets
print("\nDriver data shape:", driver_data.shape)
print("Ride data shape:", ride_data.shape)

# Check for missing values
print("\nMissing values in driver data:")
print(driver_data.isnull().sum())
print("\nMissing values in ride data:")
print(ride_data.isnull().sum())

# Convert timestamp to datetime
ride_data['timestamp'] = pd.to_datetime(ride_data['timestamp'])

# Extract hour, day of week, and month
ride_data['hour'] = ride_data['timestamp'].dt.hour
ride_data['day_of_week'] = ride_data['timestamp'].dt.dayofweek
ride_data['month'] = ride_data['timestamp'].dt.month

# Identify peak hours based on ride frequency
hourly_rides = ride_data.groupby('hour').size()
plt.figure(figsize=(12, 6))
hourly_rides.plot(kind='bar')
plt.title('Ride Distribution by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Rides')
plt.savefig('hourly_ride_distribution.png')

# Define peak hours (assuming morning and evening peaks)
morning_peak_start, morning_peak_end = 7, 10
evening_peak_start, evening_peak_end = 16, 20

# Filter rides during peak hours
peak_rides = ride_data[
    ((ride_data['hour'] >= morning_peak_start) & (ride_data['hour'] <= morning_peak_end)) |
    ((ride_data['hour'] >= evening_peak_start) & (ride_data['hour'] <= evening_peak_end))
]

print(f"\nTotal rides: {len(ride_data)}")
print(f"Peak hour rides: {len(peak_rides)} ({len(peak_rides)/len(ride_data)*100:.2f}%)")

# Create a heatmap of pickup locations during peak hours
print("\nGenerating heatmap of peak hour pickups...")
peak_map = folium.Map(location=[12.9716, 77.5946], zoom_start=12)  # Bengaluru coordinates
heat_data = [[row['pickup_lat'], row['pickup_lng']] for _, row in peak_rides.iterrows()]
HeatMap(heat_data).add_to(peak_map)
peak_map.save('peak_hour_heatmap.html')

# Identify hotspot areas based on pickup density
from sklearn.cluster import DBSCAN

# Use DBSCAN to identify clusters of pickup locations
coords = peak_rides[['pickup_lat', 'pickup_lng']].values
db = DBSCAN(eps=0.005, min_samples=50).fit(coords)
peak_rides['cluster'] = db.labels_

# Count rides in each cluster
cluster_counts = peak_rides[peak_rides['cluster'] != -1].groupby('cluster').size().reset_index(name='count')
cluster_counts = cluster_counts.sort_values('count', ascending=False)

# Get the center of each cluster
cluster_centers = peak_rides[peak_rides['cluster'] != -1].groupby('cluster')[['pickup_lat', 'pickup_lng']].mean()

# Merge counts with centers
hotspots = pd.merge(cluster_centers, cluster_counts, on='cluster')
print("\nTop hotspot areas:")
print(hotspots.head(10))
# After the line where hotspots are created (around line 80)
# Add this code to get area names for each hotspot

print("\nIdentifying area names for hotspots...")
from geopy.geocoders import Nominatim
import time
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Initialize the geocoder with a higher timeout
geolocator = Nominatim(user_agent="hotspot_prediction_app", timeout=10)

# SSL Certificate Verification Error with Nominatim Geocoding

# It looks like you're encountering an SSL certificate verification error when trying to connect to the Nominatim geocoding service. This is a common issue on macOS when Python can't verify the SSL certificates.

## Solution

# You need to modify your geocoding function to handle SSL certificate verification issues. Here are two approaches:

### Option 1: Disable SSL verification (quick but less secure)
# ```python
# Function to get area name from coordinates
def get_area_name(lat, lng):
    try:
        # Create a custom session with SSL verification disabled
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        
        # Use direct requests instead of geopy's built-in methods
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lng}&format=json&accept-language=en&addressdetails=1"
        response = requests.get(url, verify=False, headers={'User-Agent': 'hotspot_prediction_app'})
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} from Nominatim")
            return "Unknown Area"
            
        location = response.json()
        
        if not location or 'address' not in location:
            return "Unknown Area"
            
        address = location.get('address', {})
        
        # Try to get the most relevant name (suburb, neighbourhood, or road)
        for key in ['suburb', 'neighbourhood', 'road', 'city', 'county', 'state_district', 'state']:
            if key in address and address[key]:
                return address[key]
                
        # If we have an address but none of the above keys, use the display_name
        if 'display_name' in location:
            parts = location['display_name'].split(',')
            if parts and parts[0]:
                return parts[0].strip()
                
        return "Unknown Area"
    except Exception as e:
        print(f"Error getting location name for coordinates ({lat}, {lng}): {e}")
        return "Unknown Area"

# Function to get area name from coordinates with retry logic
def get_area_name_with_retry(lat, lng, max_retries=3):
    for attempt in range(max_retries):
        try:
            return get_area_name(lat, lng)
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            if attempt < max_retries - 1:
                print(f"Geocoding timed out, retrying ({attempt+1}/{max_retries})...")
                time.sleep(2)  # Wait longer between retries
            else:
                print(f"Failed to geocode after {max_retries} attempts: {e}")
                return "Unknown Area"

# Add area names to the hotspots dataframe
hotspots['area_name'] = "Unknown Area"
for idx, row in hotspots.iterrows():
    # Add a delay to avoid hitting API rate limits
    time.sleep(2)  # Increased delay to avoid rate limiting
    area_name = get_area_name_with_retry(row['pickup_lat'], row['pickup_lng'])
    hotspots.at[idx, 'area_name'] = area_name
    print(f"Cluster {row['cluster']}: {area_name}")

# Now modify the hotspot map to include area names
hotspot_map = folium.Map(location=[12.9716, 77.5946], zoom_start=12)  # Bengaluru coordinates
for _, row in hotspots.iterrows():
    folium.CircleMarker(
        location=[row['pickup_lat'], row['pickup_lng']],
        radius=row['count'] / 100,  # Scale the radius based on count
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.6,
        popup=f"{row['area_name']} (Cluster {row['cluster']}): {row['count']} rides"
    ).add_to(hotspot_map)
hotspot_map.save('hotspot_areas.html')
# Create a map with hotspot markers
hotspot_map = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
for _, row in hotspots.iterrows():
    folium.CircleMarker(
        location=[row['pickup_lat'], row['pickup_lng']],
        radius=row['count'] / 100,  # Scale the radius based on count
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.6,
        popup=f"Cluster {row['cluster']}: {row['count']} rides"
    ).add_to(hotspot_map)
hotspot_map.save('hotspot_areas.html')

# Prepare data for LSTM model
# We'll predict the number of rides in each hotspot area by hour

# Function to create time series data for LSTM
def create_time_series(data, lookback=24):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

# Prepare time series data for each hotspot
print("\nPreparing LSTM model data...")

# Get the top 5 hotspots
top_hotspots = hotspots.head(5)['cluster'].values

# Create a dictionary to store models for each hotspot
models = {}

for cluster in top_hotspots:
    # Filter rides for this cluster
    cluster_rides = peak_rides[peak_rides['cluster'] == cluster]
    
    # Create hourly time series
    hourly_counts = cluster_rides.groupby(pd.Grouper(key='timestamp', freq='h')).size()
    
    # Fill missing hours with 0
    idx = pd.date_range(hourly_counts.index.min(), hourly_counts.index.max(), freq='h')
    hourly_counts = hourly_counts.reindex(idx, fill_value=0)
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(hourly_counts.values.reshape(-1, 1))
    
    # Create sequences for LSTM
    lookback = 24  # Use 24 hours of history to predict the next hour
    X, y = create_time_series(scaled_data, lookback)
    
    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Store the model and related objects
    models[cluster] = {
        'model': model,
        'scaler': scaler,
        'history': history,
        'hourly_counts': hourly_counts
    }
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training History for Hotspot {cluster}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'training_history_cluster_{cluster}.png')
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test)
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(f'Ride Demand Prediction for Hotspot {cluster}')
    plt.xlabel('Time')
    plt.ylabel('Number of Rides')
    plt.legend()
    plt.savefig(f'predictions_cluster_{cluster}.png')

# Predict future hotspots
print("\nPredicting future hotspots...")

# Get the current date and time
now = datetime.now()

# Predict for the next 7 days
future_days = 7
future_predictions = {}

for cluster in top_hotspots:
    model_data = models[cluster]
    model = model_data['model']
    scaler = model_data['scaler']
    hourly_counts = model_data['hourly_counts']
    
    # Get the last sequence of data
    last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
    
    # Initialize predictions array
    future_hours = future_days * 24
    predictions = np.zeros(future_hours)
    
    # Predict one step at a time and use the prediction as input for the next step
    for i in range(future_hours):
        next_pred = model.predict(last_sequence)
        predictions[i] = next_pred[0, 0]
        
        # Update the sequence with the new prediction
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_pred
    
    # Inverse transform the predictions
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    # Create a date range for the predictions
    future_dates = pd.date_range(start=now, periods=future_hours, freq='h')
    
    # Store the predictions
    future_predictions[cluster] = pd.Series(predictions, index=future_dates)
    
    # Plot future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(future_predictions[cluster])
    plt.title(f'Future Ride Demand Prediction for Hotspot {cluster}')
    plt.xlabel('Date')
    plt.ylabel('Predicted Number of Rides')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'future_predictions_cluster_{cluster}.png')

# Create a comprehensive hotspot prediction dashboard
print("\nGenerating hotspot prediction dashboard...")

# Create a directory for the dashboard
os.makedirs('hotspot_dashboard', exist_ok=True)

# Create an HTML file for the dashboard
with open('hotspot_dashboard/index.html', 'w') as f:
    f.write('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bengaluru Ride Hotspot Prediction Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .section { margin-bottom: 40px; }
            .flex-container { display: flex; flex-wrap: wrap; justify-content: space-between; }
            .chart { width: 48%; margin-bottom: 20px; }
            .full-width { width: 100%; }
            iframe { border: none; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Bengaluru Ride Hotspot Prediction Dashboard</h1>
                <p>AI-driven analysis and prediction of ride demand hotspots</p>
            </div>
            
            <div class="section">
                <h2>Current Hotspot Map</h2>
                <div class="full-width">
                    <iframe src="../hotspot_areas.html" width="100%" height="500px"></iframe>
                </div>
            </div>
            
            <div class="section">
                <h2>Peak Hour Heatmap</h2>
                <div class="full-width">
                    <iframe src="../peak_hour_heatmap.html" width="100%" height="500px"></iframe>
                </div>
            </div>
            
            <div class="section">
                <h2>Hourly Ride Distribution</h2>
                <div class="full-width">
                    <img src="../hourly_ride_distribution.png" width="100%">
                </div>
            </div>
            
            <div class="section">
                <h2>Top Hotspot Areas</h2>
                <table>
                    <tr>
                        <th>Cluster</th>
                        <th>Latitude</th>
                        <th>Longitude</th>
                        <th>Ride Count</th>
                        <th>Approximate Location</th>
                    </tr>
    ''')
    
    # Add hotspot data to the table
    for _, row in hotspots.head(10).iterrows():
        f.write(f'''
                    <tr>
                        <td>{row['cluster']}</td>
                        <td>{row['pickup_lat']:.6f}</td>
                        <td>{row['pickup_lng']:.6f}</td>
                        <td>{row['count']}</td>
                        <td>{row['area_name']}</td>
                    </tr>
        ''')
    
    f.write('''
                </table>
            </div>
            
            <div class="section">
                <h2>Hotspot Predictions</h2>
    ''')
    
    # Add prediction charts for each hotspot
    for cluster in top_hotspots:
        # Get the area name for this cluster
        area_info = hotspots[hotspots['cluster'] == cluster]
        if area_info.empty:
            area_name = f"Unknown Area (Hotspot {cluster})"
        else:
            area_name = area_info['area_name'].values[0]
            if area_name == "Unknown Area" or not area_name:
                area_name = f"Unknown Area (Hotspot {cluster})"
        
        f.write(f'''
                <div class="chart">
                    <h3>{area_name} - Historical Prediction</h3>
                    <img src="../predictions_cluster_{cluster}.png" width="100%">
                </div>
                <div class="chart">
                    <h3>{area_name} - Future Prediction</h3>
                    <img src="../future_predictions_cluster_{cluster}.png" width="100%">
                </div>
        ''')
    
    f.write('''
            </div>
            
            <div class="section">
                <h2>Recommendations for Drivers</h2>
                <ul>
                    <li>Morning peak hours (7-10 AM): Position yourself near residential areas and major tech parks</li>
                    <li>Evening peak hours (4-8 PM): Focus on business districts and tech parks for return trips</li>
                    <li>Weekends: Tourist spots and shopping areas show higher demand</li>
                    <li>Check the prediction dashboard daily for the most up-to-date hotspot information</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>© 2023 Namma Yatri Hackathon - AI-Powered Ride Demand Prediction</p>
            </div>
        </div>
    </body>
    </html>
    ''')

print("\nDashboard generated successfully. Open 'hotspot_dashboard/index.html' to view it.")

# Generate driver recommendations based on predictions
def generate_driver_recommendations(future_predictions, hotspots):
    """
    Generate personalized recommendations for drivers based on predicted demand
    
    Parameters:
    -----------
    future_predictions : dict
        Dictionary with cluster IDs as keys and predicted demand Series as values
    hotspots : DataFrame
        DataFrame containing hotspot information
    
    Returns:
    --------
    recommendations : dict
        Dictionary with time periods as keys and recommendation lists as values
    """
    recommendations = {
        'morning_rush': [],
        'evening_rush': [],
        'weekend': [],
        'overall': []
    }
    
    # Define time periods
    morning_hours = range(7, 11)  # 7 AM to 10 AM
    evening_hours = range(16, 21)  # 4 PM to 8 PM
    weekend_days = [5, 6]  # Saturday and Sunday
    
    # Process each hotspot's predictions
    for cluster, predictions in future_predictions.items():
        try:
            # Get hotspot location and area name
            hotspot_info = hotspots[hotspots['cluster'] == cluster].iloc[0]
            area_name = hotspot_info['area_name']
            if area_name == "Unknown Area" or not area_name:
                area_name = f"Unknown Area (Hotspot {cluster})"
            
            location = f"{area_name}, Lat: {hotspot_info['pickup_lat']:.4f}, Lng: {hotspot_info['pickup_lng']:.4f}"
            
            # Calculate average demand for different time periods
            morning_demand = predictions[predictions.index.hour.isin(morning_hours)].mean()
            evening_demand = predictions[predictions.index.hour.isin(evening_hours)].mean()
            weekend_demand = predictions[predictions.index.dayofweek.isin(weekend_days)].mean()
            overall_demand = predictions.mean()
            
            # Add to recommendations based on demand
            if morning_demand > 0:  # Changed from overall_demand * 1.2 to ensure we get some recommendations
                recommendations['morning_rush'].append((location, float(morning_demand)))
            
            if evening_demand > 0:  # Changed from overall_demand * 1.2
                recommendations['evening_rush'].append((location, float(evening_demand)))
                
            if weekend_demand > 0:  # Changed from overall_demand * 1.1
                recommendations['weekend'].append((location, float(weekend_demand)))
                
            recommendations['overall'].append((location, float(overall_demand)))
        except Exception as e:
            print(f"Error processing recommendations for cluster {cluster}: {e}")
    
    # Sort recommendations by predicted demand
    for key in recommendations:
        recommendations[key] = sorted(recommendations[key], key=lambda x: x[1], reverse=True)
    
    return recommendations

# Generate and save driver recommendations
print("\nGenerating driver recommendations...")
recommendations = generate_driver_recommendations(future_predictions, hotspots)

# Save recommendations to a text file
with open('driver_recommendations.txt', 'w') as f:
    f.write("DRIVER RECOMMENDATIONS BASED ON PREDICTED DEMAND\n")
    f.write("==============================================\n\n")
    
    f.write("MORNING RUSH HOUR (7 AM - 10 AM)\n")
    f.write("--------------------------------\n")
    if recommendations['morning_rush']:
        for i, (location, demand) in enumerate(recommendations['morning_rush'], 1):
            f.write(f"{i}. {location} - Expected rides: {demand:.1f}\n")
    else:
        f.write("No specific recommendations for this time period\n")
    
    f.write("\nEVENING RUSH HOUR (4 PM - 8 PM)\n")
    f.write("------------------------------\n")
    if recommendations['evening_rush']:
        for i, (location, demand) in enumerate(recommendations['evening_rush'], 1):
            f.write(f"{i}. {location} - Expected rides: {demand:.1f}\n")
    else:
        f.write("No specific recommendations for this time period\n")
    
    f.write("\nWEEKEND HOTSPOTS\n")
    f.write("---------------\n")
    if recommendations['weekend']:
        for i, (location, demand) in enumerate(recommendations['weekend'], 1):
            f.write(f"{i}. {location} - Expected rides: {demand:.1f}\n")
    else:
        f.write("No specific recommendations for this time period\n")
    
    f.write("\nOVERALL TOP LOCATIONS\n")
    f.write("--------------------\n")
    for i, (location, demand) in enumerate(recommendations['overall'][:5], 1):
        f.write(f"{i}. {location} - Expected rides: {demand:.1f}\n")

print("Driver recommendations saved to 'driver_recommendations.txt'")

# Calculate model performance metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("\nCalculating model performance metrics...")
metrics = {}

for cluster in top_hotspots:
    model_data = models[cluster]
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Get predictions and actual values
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test)
    
    # Calculate metrics
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)
    
    metrics[cluster] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
    }

# Print metrics
print("\nModel Performance Metrics:")
for cluster, metric in metrics.items():
    print(f"\nHotspot {cluster}:")
    print(f"  Mean Squared Error (MSE): {metric['MSE']:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {metric['RMSE']:.4f}")
    print(f"  Mean Absolute Error (MAE): {metric['MAE']:.4f}")

print("\nHotspot prediction analysis completed successfully!")