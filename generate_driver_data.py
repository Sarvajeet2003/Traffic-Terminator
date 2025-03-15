import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(43)

# Number of drivers to generate
num_drivers = 500

# Generate driver basic information
driver_ids = [f"D{np.random.randint(1000, 9999)}" for _ in range(num_drivers)]
driver_names = [f"Driver_{i}" for i in range(num_drivers)]
driver_ratings = np.random.normal(4.5, 0.3, num_drivers)
driver_ratings = np.clip(driver_ratings, 1, 5)  # Ensure ratings are between 1 and 5

# Generate driver experience (in months)
driver_experience = np.random.gamma(shape=2.0, scale=10.0, size=num_drivers)

# Generate vehicle types
vehicle_types = ["Auto", "Auto", "Auto", "Auto", "Premium Auto"]  # More weight to regular autos
driver_vehicles = [random.choice(vehicle_types) for _ in range(num_drivers)]

# Generate preferred areas (some drivers have preferences)
bengaluru_areas = [
    "Koramangala", "Indiranagar", "Whitefield", "Electronic City", "MG Road",
    "HSR Layout", "Marathahalli", "Jayanagar", "BTM Layout", "JP Nagar",
    "Bannerghatta Road", "Hebbal", "Yelahanka", "Banashankari", "Malleshwaram",
    "Rajajinagar", "Basavanagudi", "Domlur", "Bellandur", "Sarjapur Road"
]

# Some drivers have preferred areas, others don't
driver_preferred_areas = []
for _ in range(num_drivers):
    if random.random() < 0.6:  # 60% of drivers have preferred areas
        num_preferred = random.randint(1, 3)
        preferred = random.sample(bengaluru_areas, num_preferred)
        driver_preferred_areas.append(", ".join(preferred))
    else:
        driver_preferred_areas.append(None)

# Generate working patterns
working_patterns = ["Full-time", "Part-time Morning", "Part-time Evening", "Weekend Only", "Weekday Only"]
driver_working_patterns = [random.choice(working_patterns) for _ in range(num_drivers)]

# Generate acceptance rates (overall)
# More experienced drivers tend to have more strategic acceptance rates
acceptance_rates = []
for exp in driver_experience:
    if exp < 6:  # New drivers
        rate = np.random.beta(8, 2) * 100  # Higher acceptance rate
    elif exp < 24:  # Moderate experience
        rate = np.random.beta(6, 2) * 100
    else:  # Experienced drivers
        rate = np.random.beta(5, 2) * 100  # More selective
    acceptance_rates.append(rate)

# Generate peak hour acceptance rates (usually lower than overall)
peak_acceptance_rates = []
for rate in acceptance_rates:
    # Peak hour acceptance is typically lower
    peak_rate = rate * np.random.uniform(0.7, 0.9)  # 70-90% of normal rate
    peak_acceptance_rates.append(peak_rate)

# Generate long-distance acceptance rates (usually lower than overall)
long_distance_acceptance_rates = []
for rate in acceptance_rates:
    # Long distance acceptance is typically lower
    long_rate = rate * np.random.uniform(0.6, 0.85)  # 60-85% of normal rate
    long_distance_acceptance_rates.append(long_rate)

# Generate average daily earnings
daily_earnings = []
for pattern in driver_working_patterns:
    if pattern == "Full-time":
        base_earning = np.random.normal(1200, 200)
    elif pattern in ["Part-time Morning", "Part-time Evening"]:
        base_earning = np.random.normal(700, 150)
    else:  # Weekend or Weekday only
        base_earning = np.random.normal(900, 180)
    daily_earnings.append(max(500, base_earning))  # Ensure minimum earnings

# Generate average daily trips
daily_trips = []
for earning in daily_earnings:
    # Roughly calculate trips based on earnings
    # Assuming average trip earns 150-200 INR
    avg_trip_earning = np.random.uniform(150, 200)
    trips = earning / avg_trip_earning
    daily_trips.append(round(trips, 1))

# Generate driver home locations
driver_home_areas = []
driver_home_lats = []
driver_home_lngs = []

# Define Bengaluru neighborhoods with their approximate coordinates
bengaluru_coords = {
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
    "Sarjapur Road": (12.9102, 77.6870)
}

for _ in range(num_drivers):
    area = random.choice(list(bengaluru_coords.keys()))
    lat, lng = bengaluru_coords[area]
    # Add some noise to coordinates to spread drivers around
    lat += np.random.normal(0, 0.005)
    lng += np.random.normal(0, 0.005)
    
    driver_home_areas.append(area)
    driver_home_lats.append(lat)
    driver_home_lngs.append(lng)

# Generate driver active hours
active_start_hours = []
active_end_hours = []

for pattern in driver_working_patterns:
    if pattern == "Full-time":
        start = np.random.randint(6, 10)
        end = start + np.random.randint(8, 12)
        if end > 24:
            end = 24
    elif pattern == "Part-time Morning":
        start = np.random.randint(6, 9)
        end = start + np.random.randint(4, 7)
    elif pattern == "Part-time Evening":
        start = np.random.randint(15, 18)
        end = start + np.random.randint(4, 7)
        if end > 24:
            end = 24
    elif pattern == "Weekend Only":
        start = np.random.randint(8, 12)
        end = start + np.random.randint(6, 10)
        if end > 24:
            end = 24
    else:  # Weekday Only
        start = np.random.randint(7, 10)
        end = start + np.random.randint(7, 11)
        if end > 24:
            end = 24
    
    active_start_hours.append(start)
    active_end_hours.append(end)

# Generate driver cancellation rates
cancellation_rates = []
for exp in driver_experience:
    if exp < 6:  # New drivers
        rate = np.random.beta(2, 8) * 100  # Lower cancellation rate
    elif exp < 24:  # Moderate experience
        rate = np.random.beta(2, 10) * 100
    else:  # Experienced drivers
        rate = np.random.beta(1.5, 12) * 100  # Even lower cancellation rate
    cancellation_rates.append(rate)

# Generate driver incentive eligibility (based on ratings and experience)
incentive_eligible = []
for rating, exp in zip(driver_ratings, driver_experience):
    if rating >= 4.5 and exp >= 3:
        incentive_eligible.append(True)
    elif rating >= 4.0 and exp >= 6:
        incentive_eligible.append(True)
    elif rating >= 3.8 and exp >= 12:
        incentive_eligible.append(True)
    else:
        incentive_eligible.append(False)

# Generate driver registration dates
registration_dates = []
current_date = datetime(2023, 9, 1)
for exp in driver_experience:
    # Convert experience from months to days
    days_ago = int(exp * 30)
    reg_date = current_date - timedelta(days=days_ago)
    registration_dates.append(reg_date)

# Generate driver language preferences
languages = ["Kannada", "English", "Hindi", "Tamil", "Telugu"]
driver_languages = []
for _ in range(num_drivers):
    num_languages = np.random.randint(1, 4)
    driver_langs = random.sample(languages, num_languages)
    driver_languages.append(", ".join(driver_langs))

# Create DataFrame
driver_data = {
    "driver_id": driver_ids,
    "name": driver_names,
    "rating": [round(r, 2) for r in driver_ratings],
    "experience_months": [round(e, 1) for e in driver_experience],
    "vehicle_type": driver_vehicles,
    "preferred_areas": driver_preferred_areas,
    "working_pattern": driver_working_patterns,
    "acceptance_rate": [round(r, 2) for r in acceptance_rates],
    "peak_hour_acceptance_rate": [round(r, 2) for r in peak_acceptance_rates],
    "long_distance_acceptance_rate": [round(r, 2) for r in long_distance_acceptance_rates],
    "cancellation_rate": [round(r, 2) for r in cancellation_rates],
    "avg_daily_earnings": [round(e, 2) for e in daily_earnings],
    "avg_daily_trips": daily_trips,
    "home_area": driver_home_areas,
    "home_lat": [round(l, 6) for l in driver_home_lats],
    "home_lng": [round(l, 6) for l in driver_home_lngs],
    "active_start_hour": active_start_hours,
    "active_end_hour": active_end_hours,
    "incentive_eligible": incentive_eligible,
    "registration_date": registration_dates,
    "languages": driver_languages
}

# Convert to DataFrame and save
drivers_df = pd.DataFrame(driver_data)
drivers_df.to_csv('/Users/sarvajeethuk/Desktop/Synthetic/driver_data.csv', index=False)
print(f"Generated data for {num_drivers} drivers")