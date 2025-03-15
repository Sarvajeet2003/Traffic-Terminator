import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid

# Set random seed for reproducibility
np.random.seed(42)

# Define Bengaluru neighborhoods with their approximate coordinates
bengaluru_areas = {
    "Koramangala": (12.9352, 77.6245),
    "Indiranagar": (12.9784, 77.6408),
    "Whitefield": (12.9698, 77.7499),
    "Electronic City": (12.8399, 77.6770),
    "MG Road": (12.9756, 77.6097),
    "HSR Layout": (12.9116, 77.6474),
    "Marathahalli": (12.9591, 77.6974),
    "Jayanagar": (12.9299, 77.5933),
    "BTM Layout": (12.9166, 77.6101),
    "JP Nagar": (12.9077, 77.5906),
    "Bannerghatta Road": (12.8933, 77.5978),
    "Hebbal": (13.0358, 77.5970),
    "Yelahanka": (13.1005, 77.5963),
    "Banashankari": (12.9255, 77.5468),
    "Malleshwaram": (13.0035, 77.5709),
    "Rajajinagar": (12.9866, 77.5517),
    "Basavanagudi": (12.9422, 77.5760),
    "Domlur": (12.9609, 77.6387),
    "Bellandur": (12.9257, 77.6649),
    "Sarjapur Road": (12.9102, 77.6870),
    "Kengeri": (12.9054, 77.4820),
    "Peenya": (13.0268, 77.5364),
    "Yeshwanthpur": (13.0279, 77.5409),
    "CV Raman Nagar": (12.9850, 77.6606),
    "Banaswadi": (13.0167, 77.6461),
    "Kalyan Nagar": (13.0279, 77.6394),
    "ITPL": (12.9656, 77.7479),
    "Majestic": (12.9767, 77.5713),
    "Shivajinagar": (12.9863, 77.6047),
    "Richmond Town": (12.9684, 77.6099),
    "Cubbon Park":(12.9809, 77.6105)
}

# Define peak hours and their probability weights
time_slots = {
    "early_morning": (5, 8, 0.7),  # 5 AM to 8 AM, moderate demand
    "morning_peak": (8, 10, 1.5),   # 8 AM to 10 AM, high demand
    "mid_day": (10, 16, 0.6),       # 10 AM to 4 PM, lower demand
    "evening_peak": (16, 20, 1.8),  # 4 PM to 8 PM, highest demand
    "night": (20, 23, 0.8),         # 8 PM to 11 PM, moderate demand
    "late_night": (23, 5, 0.3)      # 11 PM to 5 AM, low demand
}

# Define weekday vs weekend demand patterns
weekday_factor = 1.2  # Higher demand on weekdays
weekend_factor = 0.9  # Lower demand on weekends

# Define weather conditions and their impact on demand
weather_conditions = {
    "Clear": 1.0,
    "Cloudy": 0.95,
    "Light Rain": 1.2,
    "Heavy Rain": 1.5,
    "Thunderstorm": 1.7
}

# Define special events that might affect demand
special_events = [
    {"name": "Tech Conference", "date": "2023-09-15", "areas": ["Whitefield", "ITPL"], "factor": 1.8},
    {"name": "Cricket Match", "date": "2023-09-20", "areas": ["MG Road", "Shivajinagar"], "factor": 2.0},
    {"name": "Music Festival", "date": "2023-09-25", "areas": ["Electronic City"], "factor": 1.7},
    {"name": "College Fest", "date": "2023-09-10", "areas": ["Jayanagar", "JP Nagar"], "factor": 1.5},
    {"name": "Marathon", "date": "2023-09-05", "areas": ["Cubbon Park", "MG Road", "Indiranagar"], "factor": 1.9}
]

# Generate ride data for 30 days
start_date = datetime(2023, 9, 1)
end_date = datetime(2023, 9, 30)
current_date = start_date

rides_data = []

while current_date <= end_date:
    # Determine if it's a weekday or weekend
    is_weekend = current_date.weekday() >= 5
    day_factor = weekend_factor if is_weekend else weekday_factor
    
    # Randomly select weather for the day
    daily_weather = random.choice(list(weather_conditions.keys()))
    weather_factor = weather_conditions[daily_weather]
    
    # Check for special events
    event_factor = 1.0
    event_name = None
    for event in special_events:
        if event["date"] == current_date.strftime("%Y-%m-%d"):
            event_factor = event["factor"]
            event_name = event["name"]
            event_areas = event["areas"]
            break
    
    # Generate rides for each hour of the day
    for hour in range(24):
        # Determine time slot and its demand factor
        time_factor = 0.5  # Default low demand
        for slot, (start_hour, end_hour, factor) in time_slots.items():
            if start_hour <= hour < end_hour or (start_hour > end_hour and (hour >= start_hour or hour < end_hour)):
                time_factor = factor
                break
        
        # Calculate base number of rides for this hour
        base_rides = int(np.random.poisson(50 * time_factor * day_factor * weather_factor * event_factor))
        
        # Generate individual rides
        for _ in range(base_rides):
            # Generate pickup location
            if event_name and random.random() < 0.7:  # 70% chance of pickup near event
                pickup_area = random.choice(event_areas)
            else:
                pickup_area = random.choice(list(bengaluru_areas.keys()))
            
            pickup_lat, pickup_lng = bengaluru_areas[pickup_area]
            # Add some noise to coordinates
            pickup_lat += np.random.normal(0, 0.003)
            pickup_lng += np.random.normal(0, 0.003)
            
            # Generate drop location (different from pickup)
            drop_areas = list(bengaluru_areas.keys())
            if pickup_area in drop_areas:
                drop_areas.remove(pickup_area)
            drop_area = random.choice(drop_areas)
            
            drop_lat, drop_lng = bengaluru_areas[drop_area]
            # Add some noise to coordinates
            drop_lat += np.random.normal(0, 0.003)
            drop_lng += np.random.normal(0, 0.003)
            
            # Calculate approximate distance (in km)
            # Using simplified formula for demonstration
            distance = np.sqrt((pickup_lat - drop_lat)**2 + (pickup_lng - drop_lng)**2) * 111
            
            # Generate timestamp
            minute = np.random.randint(0, 60)
            timestamp = current_date.replace(hour=hour, minute=minute)
            
            # Calculate fare (base fare + distance fare + time of day premium)
            base_fare = 30  # Base fare in INR
            distance_fare = distance * 12  # INR per km
            time_premium = 1.0
            if 8 <= hour < 10 or 17 <= hour < 20:  # Peak hours
                time_premium = 1.3
            
            estimated_fare = base_fare + distance_fare * time_premium
            
            # Determine if the ride was accepted or denied
            # Factors affecting denial: distance, time of day, weather
            denial_probability = 0.05  # Base denial rate
            
            # Long distance trips more likely to be denied
            if distance > 15:
                denial_probability += 0.15
            elif distance > 10:
                denial_probability += 0.1
            elif distance > 5:
                denial_probability += 0.05
            
            # Peak hours have higher denial rates
            if 8 <= hour < 10 or 17 <= hour < 20:
                denial_probability += 0.1
            
            # Bad weather increases denials
            if daily_weather in ["Heavy Rain", "Thunderstorm"]:
                denial_probability += 0.15
            
            # Generate wait time (affected by time of day and weather)
            base_wait_time = np.random.gamma(shape=2.0, scale=2.0)  # Base distribution
            wait_time_multiplier = 1.0
            
            # Peak hours have longer wait times
            if 8 <= hour < 10 or 17 <= hour < 20:
                wait_time_multiplier *= 1.5
            
            # Bad weather increases wait times
            if daily_weather in ["Heavy Rain", "Thunderstorm"]:
                wait_time_multiplier *= 1.7
            elif daily_weather == "Light Rain":
                wait_time_multiplier *= 1.3
            
            wait_time = base_wait_time * wait_time_multiplier
            
            # Determine status
            if random.random() < denial_probability:
                status = "Denied"
                driver_id = None
                actual_fare = 0
                completion_time = None
            else:
                status = "Completed"
                driver_id = f"D{np.random.randint(1000, 9999)}"
                
                # Actual fare might differ slightly from estimated
                actual_fare = estimated_fare * np.random.uniform(0.95, 1.05)
                
                # Calculate trip duration based on distance and time of day
                # Assume average speed of 20 km/h during peak, 30 km/h otherwise
                speed = 20 if (8 <= hour < 10 or 17 <= hour < 20) else 30
                duration_hours = distance / speed
                duration_minutes = duration_hours * 60
                
                # Add some randomness to duration
                duration_minutes *= np.random.uniform(0.9, 1.2)
                
                # Calculate completion time
                completion_time = timestamp + timedelta(minutes=duration_minutes)
            
            # Create ride record
            ride = {
                "ride_id": str(uuid.uuid4()),
                "timestamp": timestamp,
                "pickup_area": pickup_area,
                "pickup_lat": pickup_lat,
                "pickup_lng": pickup_lng,
                "drop_area": drop_area,
                "drop_lat": drop_lat,
                "drop_lng": drop_lng,
                "distance_km": round(distance, 2),
                "estimated_fare": round(estimated_fare, 2),
                "actual_fare": round(actual_fare, 2),
                "wait_time_minutes": round(wait_time, 2),
                "status": status,
                "driver_id": driver_id,
                "completion_time": completion_time,
                "weather": daily_weather,
                "is_weekend": is_weekend,
                "special_event": event_name
            }
            
            rides_data.append(ride)
    
    # Move to next day
    current_date += timedelta(days=1)

# Convert to DataFrame and save
rides_df = pd.DataFrame(rides_data)
rides_df.to_csv('/Users/sarvajeethuk/Desktop/Synthetic/ride_data.csv', index=False)
print(f"Generated {len(rides_df)} ride records")