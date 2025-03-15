import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import time
import pickle
import random

# Set page configuration
st.set_page_config(
    page_title="Namma Yatri - Real-Life Scenario",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5733;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3366FF;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .scenario-step {
        background-color: rgba(240, 242, 246, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #3366FF;
        margin-bottom: 1.5rem;
    }
    .rider-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #FF5733;
        margin-bottom: 1rem;
    }
    .driver-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #33FF57;
        margin-bottom: 1rem;
    }
    .system-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #9933FF;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.2);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FF5733;
    }
    .metric-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
    }
    .notification {
        background-color: rgba(51, 102, 255, 0.1);
        padding: 0.8rem;
        border-radius: 0.5rem;
        border: 1px solid #3366FF;
        margin-bottom: 1rem;
    }
    .time-display {
        font-size: 1.5rem;
        font-weight: bold;
        color: #33FF57;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load pricing model
@st.cache_resource
def load_pricing_model():
    try:
        with open('dynamic_pricing_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        # Return dummy model data if file not found
        return {
            'q_table': np.random.random((5, 5, 24, 2, 3, 11)),
            'base_fare': 50,
            'price_multipliers': np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        }

# Function to predict price
def predict_price(demand, supply, hour, is_weekend, weather, model_data):
    # Discretize demand (assuming demand is normalized between 0 and 1)
    demand_level = min(int(demand * 5), 4)
    
    # Discretize supply (assuming supply is normalized between 0 and 1)
    supply_level = min(int(supply * 5), 4)
    
    # Day type (0 for weekday, 1 for weekend)
    day_type = 1 if is_weekend else 0
    
    # Weather condition
    if weather == 'good':
        weather_condition = 0
    elif weather == 'moderate':
        weather_condition = 1
    else:  # bad
        weather_condition = 2
    
    # Get state representation
    state = (demand_level, supply_level, hour, day_type, weather_condition)
    
    # Choose best action (no exploration during prediction)
    q_values = model_data['q_table'][state]
    action = np.argmax(q_values)
    
    # Return corresponding price multiplier
    return model_data['price_multipliers'][action]

# Get Bengaluru locations
def get_bengaluru_locations():
    return {
        "indiranagar": {"lat": 12.9784, "lng": 77.6408},
        "electronic city": {"lat": 12.8399, "lng": 77.6770},
        "marathahalli": {"lat": 12.9591, "lng": 77.6974},
        "koramangala": {"lat": 12.9352, "lng": 77.6245},
    }

# Main function
def main():
    # Load model data
    model_data = load_pricing_model()
    locations = get_bengaluru_locations()
    
    # Sidebar
    st.sidebar.image("image.png", width=200)
    st.sidebar.title("Scenario Controls")
    
    # Add a reset button
    if st.sidebar.button("Reset Scenario"):
        st.session_state.step = 0
        st.session_state.auto_play = False
    
    # Auto-play toggle
    if 'auto_play' not in st.session_state:
        st.session_state.auto_play = False
    
    st.session_state.auto_play = st.sidebar.checkbox("Auto-play scenario", value=st.session_state.auto_play)
    
    # Speed control for auto-play
    auto_play_speed = st.sidebar.slider("Auto-play speed", 1, 10, 5)
    
    # Initialize step if not in session state
    if 'step' not in st.session_state:
        st.session_state.step = 0
    
    # Main content
    st.markdown("<h1 class='main-header'>Morning Rush in Bengaluru: A Day with Namma Yatri</h1>", unsafe_allow_html=True)
    
    # Scenario steps
    steps = [
        "7:15 AM - Priya opens the Namma Yatri app",
        "7:16 AM - Rajesh receives driver notification",
        "7:18 AM - Matching process begins",
        "7:20 AM - Ride sharing option appears",
        "7:25 AM - Queue management in action",
        "8:45 AM - Outcome analysis"
    ]
    
    # Display progress
    progress = st.progress(st.session_state.step / (len(steps) - 1))
    
    # Display current time based on step
    times = ["7:15 AM", "7:16 AM", "7:18 AM", "7:20 AM", "7:25 AM", "8:45 AM"]
    st.markdown(f"<div class='time-display'>{times[st.session_state.step]}</div>", unsafe_allow_html=True)
    
    # Step navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("Previous Step") and st.session_state.step > 0:
            st.session_state.step -= 1
            st.rerun()
    
    with col3:
        if st.button("Next Step") and st.session_state.step < len(steps) - 1:
            st.session_state.step += 1
            st.rerun()
    
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>{steps[st.session_state.step]}</h3>", unsafe_allow_html=True)
    
    # Display scenario content based on current step
    if st.session_state.step == 0:
        # Step 1: Priya opens the app
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='rider-card'>", unsafe_allow_html=True)
            st.markdown("### Priya - Software Engineer")
            st.markdown("**Location:** Indiranagar")
            st.markdown("**Destination:** Electronic City")
            st.markdown("**Needs to arrive by:** 9:00 AM")
            st.markdown("**Namma Yatri Plus subscriber:** Yes")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='system-card'>", unsafe_allow_html=True)
            st.markdown("### AI System Analysis")
            st.markdown("**LSTM Hotspot Prediction:**")
            st.markdown("- Indiranagar identified as high-demand area (87% confidence)")
            st.markdown("- Predicted 28 drivers in the area")
            st.markdown("- Morning peak hour pattern detected")
            
            st.markdown("**Dynamic Pricing Evaluation:**")
            st.markdown("- Demand level: High (0.8)")
            st.markdown("- Supply level: Medium (0.5)")
            st.markdown("- Time: 7:15 AM (peak hour)")
            st.markdown("- Day type: Weekday")
            st.markdown("- Weather: Light rain")
            
            # Calculate price multiplier
            multiplier = predict_price(0.8, 0.5, 7, False, 'moderate', model_data)
            base_fare = 150
            final_fare = base_fare * multiplier
            
            st.markdown(f"**Price Multiplier:** {multiplier:.1f}x")
            st.markdown(f"**Estimated Fare:** ₹{final_fare:.0f}")
            st.markdown(f"**Estimated Wait Time:** 6 minutes")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Create a map centered on Indiranagar
            m = folium.Map(location=[locations["indiranagar"]["lat"], locations["indiranagar"]["lng"]], zoom_start=12)
            
            # Add marker for Priya's location
            folium.Marker(
                [locations["indiranagar"]["lat"], locations["indiranagar"]["lng"]],
                popup="Priya's Location (Indiranagar)",
                icon=folium.Icon(color="red", icon="user")
            ).add_to(m)
            
            # Add marker for destination
            folium.Marker(
                [locations["electronic city"]["lat"], locations["electronic city"]["lng"]],
                popup="Destination (Electronic City)",
                icon=folium.Icon(color="green", icon="flag")
            ).add_to(m)
            
            # Add line connecting origin and destination
            folium.PolyLine(
                locations=[
                    [locations["indiranagar"]["lat"], locations["indiranagar"]["lng"]],
                    [locations["electronic city"]["lat"], locations["electronic city"]["lng"]]
                ],
                color="blue",
                weight=3,
                opacity=0.7,
                dash_array="10"
            ).add_to(m)
            
            # Display the map
            folium_static(m)
            
            # App notification
            st.markdown("<div class='notification'>", unsafe_allow_html=True)
            st.markdown("**Namma Yatri App Notification:**")
            st.markdown("Prices are slightly higher due to morning rush hour and light rain. 28 drivers are currently in your area.")
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif st.session_state.step == 1:
        # Step 2: Rajesh receives notification
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='driver-card'>", unsafe_allow_html=True)
            st.markdown("### Rajesh - Auto Driver")
            st.markdown("**Rating:** 4.8 ⭐")
            st.markdown("**Experience:** 3 years on platform")
            st.markdown("**Current Location:** Near Domlur (2.5 km from Indiranagar)")
            st.markdown("**Preferences:** Morning shifts, longer rides")
            st.markdown("**Elite Driver Status:** Silver")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='system-card'>", unsafe_allow_html=True)
            st.markdown("### Driver Recommendation System")
            st.markdown("**Analysis:**")
            st.markdown("- Driver typically works 6 AM - 2 PM shift")
            st.markdown("- Historical data shows preference for rides > 5 km")
            st.markdown("- 85% acceptance rate during peak hours")
            st.markdown("- Frequently serves Indiranagar area")
            
            st.markdown("**Personalized Recommendations:**")
            st.markdown("1. Position near Indiranagar Metro Station")
            st.markdown("2. Accept longer rides during morning peak")
            st.markdown("3. Potential for 30% higher earnings with surge pricing")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Create a map showing driver location
            m = folium.Map(location=[12.9700, 77.6200], zoom_start=13)
            
            # Add marker for Rajesh's location
            folium.Marker(
                [12.9609, 77.6387],  # Domlur coordinates
                popup="Rajesh's Location (Domlur)",
                icon=folium.Icon(color="green", icon="car")
            ).add_to(m)
            
            # Add marker for high demand area
            folium.CircleMarker(
                [locations["indiranagar"]["lat"], locations["indiranagar"]["lng"]],
                radius=300,
                popup="High Demand Area (Indiranagar)",
                color="#FF5733",
                fill=True,
                fill_color="#FF5733",
                fill_opacity=0.4
            ).add_to(m)
            
            # Add direction arrow
            folium.PolyLine(
                locations=[
                    [12.9609, 77.6387],  # Domlur
                    [locations["indiranagar"]["lat"], locations["indiranagar"]["lng"]]
                ],
                color="green",
                weight=3,
                opacity=0.7,
                arrow_style="->",
                arrow_size=12
            ).add_to(m)
            
            # Display the map
            folium_static(m)
            
            # App notification
            st.markdown("<div class='notification'>", unsafe_allow_html=True)
    
    elif st.session_state.step == 2:
        # Step 3: Matching Process
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='rider-card'>", unsafe_allow_html=True)
            st.markdown("### Ride Request Details")
            st.markdown("**Rider:** Priya (4.9★)")
            st.markdown("**Pickup:** Indiranagar")
            st.markdown("**Destination:** Electronic City (18.5 km)")
            st.markdown("**Fare:** ₹210 (1.4x surge)")
            st.markdown("**Subscription:** Namma Yatri Plus")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='system-card'>", unsafe_allow_html=True)
            st.markdown("### Matching Algorithm")
            st.markdown("**Queue Priority Factors:**")
            st.markdown("- Subscription status: +2 priority points")
            st.markdown("- Loyalty (1+ year user): +1 priority point")
            st.markdown("- Wait time: 0 minutes (just requested)")
            st.markdown("- Total priority score: 3")
            
            st.markdown("**Available Drivers:**")
            st.markdown("1. Rajesh (2.5 km away, 4.8★)")
            st.markdown("2. Sunil (3.2 km away, 4.7★)")
            st.markdown("3. Venkat (1.8 km away, 4.5★)")
            
            st.markdown("**Match Decision:**")
            st.markdown("Rajesh selected based on:")
            st.markdown("- Preference for longer rides")
            st.markdown("- High acceptance rate (85%)")
            st.markdown("- Good proximity to pickup point")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Create a map showing matching process
            m = folium.Map(location=[12.9700, 77.6200], zoom_start=12)
            
            # Add marker for Priya's location
            folium.Marker(
                [locations["indiranagar"]["lat"], locations["indiranagar"]["lng"]],
                popup="Priya's Location (Indiranagar)",
                icon=folium.Icon(color="red", icon="user")
            ).add_to(m)
            
            # Add markers for available drivers
            folium.Marker(
                [12.9609, 77.6387],  # Domlur - Rajesh
                popup="Rajesh (2.5 km away)",
                icon=folium.Icon(color="green", icon="car")
            ).add_to(m)
            
            folium.Marker(
                [12.9716, 77.6099],  # Sunil
                popup="Sunil (3.2 km away)",
                icon=folium.Icon(color="gray", icon="car")
            ).add_to(m)
            
            folium.Marker(
                [12.9850, 77.6406],  # Venkat
                popup="Venkat (1.8 km away)",
                icon=folium.Icon(color="gray", icon="car")
            ).add_to(m)
            
            # Add connecting line for matched driver
            folium.PolyLine(
                locations=[
                    [12.9609, 77.6387],  # Rajesh
                    [locations["indiranagar"]["lat"], locations["indiranagar"]["lng"]]  # Priya
                ],
                color="green",
                weight=4,
                opacity=0.8
            ).add_to(m)
            
            # Display the map
            folium_static(m)
            
            # System notification
            st.markdown("<div class='notification'>", unsafe_allow_html=True)
            st.markdown("**System Notification:**")
            st.markdown("Match confirmed! Rajesh is heading to pick up Priya. Estimated arrival in 7 minutes.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Driver app view
            st.markdown("<div class='notification'>", unsafe_allow_html=True)
            st.markdown("**Driver App View:**")
            st.markdown("New ride request: Indiranagar → Electronic City (18.5 km)")
            st.markdown("Fare: ₹210 (includes 1.4x surge)")
            st.markdown("Estimated trip time: 45 minutes")
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif st.session_state.step == 3:
        # Step 4: Ride Sharing Option
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='rider-card'>", unsafe_allow_html=True)
            st.markdown("### Ride Sharing Opportunity")
            st.markdown("**Current Rider:** Priya")
            st.markdown("**Route:** Indiranagar → Electronic City")
            st.markdown("**Current Fare:** ₹210")
            st.markdown("**Driver:** Rajesh (on the way)")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='rider-card'>", unsafe_allow_html=True)
            st.markdown("### Potential Co-Rider")
            st.markdown("**Name:** Arun (4.7★)")
            st.markdown("**Pickup:** Indiranagar")
            st.markdown("**Destination:** Marathahalli")
            st.markdown("**On the way to:** Electronic City")
            st.markdown("**Distance overlap:** 65%")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='system-card'>", unsafe_allow_html=True)
            st.markdown("### Ride Sharing Algorithm")
            st.markdown("**Analysis:**")
            st.markdown("- Route overlap: 65%")
            st.markdown("- Additional pickup time: 0 minutes (same location)")
            st.markdown("- Additional drop-off time: 8 minutes")
            st.markdown("- Total detour time: 8 minutes")
            st.markdown("- Compatibility score: 85%")
            
            st.markdown("**Fare Calculation:**")
            st.markdown("- Original fare (Priya): ₹210")
            st.markdown("- Shared ride discount: 33%")
            st.markdown("- New fare (Priya): ₹140")
            st.markdown("- Fare for Arun: ₹120")
            st.markdown("- Driver total earnings: ₹260 (24% increase)")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Create a map showing ride sharing route
            m = folium.Map(location=[12.9300, 77.6300], zoom_start=11)
            
            # Add markers for locations
            folium.Marker(
                [locations["indiranagar"]["lat"], locations["indiranagar"]["lng"]],
                popup="Pickup: Priya & Arun (Indiranagar)",
                icon=folium.Icon(color="red", icon="user")
            ).add_to(m)
            
            folium.Marker(
                [locations["marathahalli"]["lat"], locations["marathahalli"]["lng"]],
                popup="Dropoff: Arun (Marathahalli)",
                icon=folium.Icon(color="blue", icon="flag")
            ).add_to(m)
            
            folium.Marker(
                [locations["electronic city"]["lat"], locations["electronic city"]["lng"]],
                popup="Dropoff: Priya (Electronic City)",
                icon=folium.Icon(color="green", icon="flag")
            ).add_to(m)
            
            # Add route lines
            # Direct route to Electronic City
            folium.PolyLine(
                locations=[
                    [locations["indiranagar"]["lat"], locations["indiranagar"]["lng"]],
                    [locations["electronic city"]["lat"], locations["electronic city"]["lng"]]
                ],
                color="gray",
                weight=2,
                opacity=0.5,
                dash_array="5",
                popup="Original direct route"
            ).add_to(m)
            
            # Shared route
            folium.PolyLine(
                locations=[
                    [locations["indiranagar"]["lat"], locations["indiranagar"]["lng"]],
                    [locations["marathahalli"]["lat"], locations["marathahalli"]["lng"]],
                    [locations["electronic city"]["lat"], locations["electronic city"]["lng"]]
                ],
                color="green",
                weight=4,
                opacity=0.8,
                popup="Shared ride route"
            ).add_to(m)
            
            # Display the map
            folium_static(m)
            
            # App notification
            st.markdown("<div class='notification'>", unsafe_allow_html=True)
            st.markdown("**Namma Yatri App Notification:**")
            st.markdown("Share your ride with Arun and save ₹70 (33% discount). This will add approximately 8 minutes to your journey.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Decision buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Accept Shared Ride"):
                    st.success("Shared ride accepted! Your fare is now ₹140.")
            with col2:
                if st.button("Keep Private Ride"):
                    st.info("You chose to keep your private ride. Your fare remains ₹210.")
    
    elif st.session_state.step == 4:
        # Step 5: Queue Management in Koramangala
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='system-card'>", unsafe_allow_html=True)
            st.markdown("### Queue Management System")
            st.markdown("**Location:** Koramangala (Hotspot)")
            st.markdown("**Current Status:**")
            st.markdown("- Active ride requests: 45")
            st.markdown("- Available drivers: 28")
            st.markdown("- Driver deficit: 17")
            st.markdown("- Average wait time: 12 minutes")
            
            st.markdown("**Queue Priority Distribution:**")
            st.markdown("- High priority (Plus subscribers): 18 requests")
            st.markdown("- Medium priority (Loyal users): 15 requests")
            st.markdown("- Standard priority: 12 requests")
            
            st.markdown("**System Actions:**")
            st.markdown("1. Dynamic price adjustment: 1.5x → 1.7x")
            st.markdown("2. Driver notifications sent to nearby areas")
            st.markdown("3. Wait time estimates updated for users")
            st.markdown("4. Priority queue optimization")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Queue metrics
            st.markdown("### Queue Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>17</div>
                    <div class='metric-label'>Driver Deficit</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>12 min</div>
                    <div class='metric-label'>Avg. Wait Time</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>1.7x</div>
                    <div class='metric-label'>Current Surge</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>82%</div>
                    <div class='metric-label'>Acceptance Rate</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Create a map showing Koramangala hotspot
            m = folium.Map(location=[locations["koramangala"]["lat"], locations["koramangala"]["lng"]], zoom_start=13)
            
            # Add hotspot circle
            folium.CircleMarker(
                [locations["koramangala"]["lat"], locations["koramangala"]["lng"]],
                radius=500,
                popup="Koramangala Hotspot: 45 active requests",
                color="#FF5733",
                fill=True,
                fill_color="#FF5733",
                fill_opacity=0.4
            ).add_to(m)
            
            # Add driver markers (randomly placed around Koramangala)
            import random
            for i in range(28):
                lat_offset = random.uniform(-0.01, 0.01)
                lng_offset = random.uniform(-0.01, 0.01)
                folium.CircleMarker(
                    [locations["koramangala"]["lat"] + lat_offset, locations["koramangala"]["lng"] + lng_offset],
                    radius=3,
                    color="green",
                    fill=True,
                    fill_opacity=0.8
                ).add_to(m)
            
            # Add rider request markers
            for i in range(45):
                lat_offset = random.uniform(-0.015, 0.015)
                lng_offset = random.uniform(-0.015, 0.015)
                folium.CircleMarker(
                    [locations["koramangala"]["lat"] + lat_offset, locations["koramangala"]["lng"] + lng_offset],
                    radius=3,
                    color="blue",
                    fill=True,
                    fill_opacity=0.8
                ).add_to(m)
            
            # Display the map
            folium_static(m)
            
            # Queue visualization
            st.markdown("### Priority Queue Visualization")
            
            # Create a simple queue visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Data
            categories = ['High Priority', 'Medium Priority', 'Standard Priority']
            values = [18, 15, 12]
            colors = ['#FF5733', '#3366FF', '#33FF57']
            
            # Create horizontal bar chart
            bars = ax.barh(categories, values, color=colors)
            
            # Add labels and values
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                        f"{values[i]} requests", 
                        ha='left', va='center')
            
            ax.set_xlabel('Number of Requests')
            ax.set_title('Ride Requests by Priority Level')
            
            # Display the chart
            st.pyplot(fig)
    
    # App notification
            st.markdown("<div class='notification'>", unsafe_allow_html=True)
            st.markdown("**Namma Yatri App Notification:**")
            st.markdown("High demand in Indiranagar area with surge pricing. Head there for 30% higher earnings.")
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif st.session_state.step == 5:
        # Step 6: Outcome Analysis
        st.markdown("<h2 class='sub-header'>Morning Rush Hour Results</h2>", unsafe_allow_html=True)
        
        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>35%</div>
                <div class='metric-label'>Reduction in Ride Denials</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>9 min</div>
                <div class='metric-label'>Average Wait Time</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>20%</div>
                <div class='metric-label'>Increase in Driver Earnings</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>25%</div>
                <div class='metric-label'>Reduction in Driver Idle Time</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Rider outcomes
        st.markdown("<h3 class='sub-header'>Rider Outcomes</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='rider-card'>", unsafe_allow_html=True)
            st.markdown("### Priya's Journey")
            st.markdown("**Original Fare:** ₹210")
            st.markdown("**Final Fare:** ₹140 (33% savings with ride sharing)")
            st.markdown("**Wait Time:** 7 minutes (vs. avg 15 min last week)")
            st.markdown("**Arrival Time:** 8:42 AM (18 minutes before deadline)")
            st.markdown("**Satisfaction Rating:** 4.8/5")
            st.markdown("**Feedback:** 'The ride sharing option saved me money and the driver was excellent!'")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Create a simple chart for wait time comparison
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Data
            categories = ['Previous Average', 'Today (with AI Solution)']
            values = [15, 7]
            colors = ['#FF5733', '#33FF57']
            
            # Create bar chart
            bars = ax.bar(categories, values, color=colors)
            
            # Add labels and values
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f"{height} min", ha='center', va='bottom')
            
            ax.set_ylabel('Wait Time (minutes)')
            ax.set_title('Rider Wait Time Comparison')
            ax.set_ylim(0, 20)
            
            # Display the chart
            st.pyplot(fig)
        
        with col2:
            # Create a pie chart for ride sharing benefits
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Data
            labels = ['Fare Savings', 'Carbon Reduction', 'Traffic Reduction', 'Increased Availability']
            sizes = [33, 25, 22, 20]
            colors = ['#FF5733', '#33FF57', '#3366FF', '#FFFF33']
            explode = (0.1, 0, 0, 0)
            
            # Create pie chart
            ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
            ax.axis('equal')
            ax.set_title('Ride Sharing Benefits')
            
            # Display the chart
            st.pyplot(fig)
            
            st.markdown("<div class='notification'>", unsafe_allow_html=True)
            st.markdown("**System Learning:**")
            st.markdown("- Rider preference for shared rides during peak hours recorded")
            st.markdown("- Route popularity between Indiranagar and Electronic City updated")
            st.markdown("- Fare elasticity model updated based on acceptance rate")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Driver outcomes
        st.markdown("<h3 class='sub-header'>Driver Outcomes</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='driver-card'>", unsafe_allow_html=True)
            st.markdown("### Rajesh's Morning Shift")
            st.markdown("**Rides Completed:** 3 (vs. avg 2 in previous morning shifts)")
            st.markdown("**Total Earnings:** ₹580 (20% above average)")
            st.markdown("**Idle Time:** 18 minutes (vs. avg 45 min previously)")
            st.markdown("**Fuel Consumption:** 15% less per rupee earned")
            st.markdown("**Satisfaction Rating:** 4.9/5")
            st.markdown("**Feedback:** 'The recommendations helped me find more rides and earn more!'")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Create a simple chart for earnings comparison
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Data
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday (with AI)']
            earnings = [450, 470, 430, 460, 580]
            colors = ['#3366FF', '#3366FF', '#3366FF', '#3366FF', '#33FF57']
            
            # Create bar chart
            bars = ax.bar(days, earnings, color=colors)
            
            # Add labels and values
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f"₹{height}", ha='center', va='bottom')
            
            ax.set_ylabel('Earnings (₹)')
            ax.set_title('Driver Earnings Comparison (Morning Shift)')
            ax.set_ylim(0, 650)
            
            # Display the chart
            st.pyplot(fig)
        
        with col2:
            # Create a map showing completed rides
            m = folium.Map(location=[12.9300, 77.6300], zoom_start=11)
            
            # Add markers for key locations
            folium.Marker(
                [locations["indiranagar"]["lat"], locations["indiranagar"]["lng"]],
                popup="Indiranagar",
                icon=folium.Icon(color="blue")
            ).add_to(m)
            
            folium.Marker(
                [locations["marathahalli"]["lat"], locations["marathahalli"]["lng"]],
                popup="Marathahalli",
                icon=folium.Icon(color="blue")
            ).add_to(m)
            
            folium.Marker(
                [locations["electronic city"]["lat"], locations["electronic city"]["lng"]],
                popup="Electronic City",
                icon=folium.Icon(color="blue")
            ).add_to(m)
            
            folium.Marker(
                [locations["koramangala"]["lat"], locations["koramangala"]["lng"]],
                popup="Koramangala",
                icon=folium.Icon(color="blue")
            ).add_to(m)
            
            # Add route lines for completed rides
            # Ride 1: Indiranagar to Electronic City (with Marathahalli detour)
            folium.PolyLine(
                locations=[
                    [locations["indiranagar"]["lat"], locations["indiranagar"]["lng"]],
                    [locations["marathahalli"]["lat"], locations["marathahalli"]["lng"]],
                    [locations["electronic city"]["lat"], locations["electronic city"]["lng"]]
                ],
                color="green",
                weight=4,
                opacity=0.8,
                popup="Ride 1: Shared ride to Electronic City"
            ).add_to(m)
            
            # Ride 2: Electronic City to Koramangala
            folium.PolyLine(
                locations=[
                    [locations["electronic city"]["lat"], locations["electronic city"]["lng"]],
                    [locations["koramangala"]["lat"], locations["koramangala"]["lng"]]
                ],
                color="blue",
                weight=4,
                opacity=0.8,
                popup="Ride 2: Electronic City to Koramangala"
            ).add_to(m)
            
            # Ride 3: Koramangala to Indiranagar
            folium.PolyLine(
                locations=[
                    [locations["koramangala"]["lat"], locations["koramangala"]["lng"]],
                    [locations["indiranagar"]["lat"], locations["indiranagar"]["lng"]]
                ],
                color="purple",
                weight=4,
                opacity=0.8,
                popup="Ride 3: Koramangala to Indiranagar"
            ).add_to(m)
            
            # Display the map
            folium_static(m)
            
            st.markdown("<div class='notification'>", unsafe_allow_html=True)
            st.markdown("**System Learning:**")
            st.markdown("- Driver preference for longer rides confirmed")
            st.markdown("- Route efficiency model updated")
            st.markdown("- Driver positioning recommendations refined")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Platform outcomes
        st.markdown("<h3 class='sub-header'>Platform Outcomes</h3>", unsafe_allow_html=True)
        
        # Create a line chart for key metrics over time
        fig = plt.figure(figsize=(12, 6))
        
        # Sample data - showing improvement over 5 days
        days = list(range(1, 6))
        wait_times = [15, 14, 12, 10, 9]
        ride_denials = [100, 90, 80, 70, 65]
        driver_earnings = [100, 105, 110, 115, 120]
        
        # Create plot with dual y-axis
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        
        # Plot data
        line1 = ax1.plot(days, wait_times, 'b-', marker='o', label='Avg Wait Time (min)')
        line2 = ax1.plot(days, ride_denials, 'r-', marker='s', label='Ride Denials (%)')
        line3 = ax2.plot(days, driver_earnings, 'g-', marker='^', label='Driver Earnings (%)')
        
        # Set labels and title
        ax1.set_xlabel('Days Since Implementation')
        ax1.set_ylabel('Wait Time / Ride Denials')
        ax2.set_ylabel('Driver Earnings (% of baseline)')
        ax1.set_title('Key Metrics Over Time')
        
        # Add legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center')
        
        # Set x-axis ticks
        ax1.set_xticks(days)
        
        # Display the chart
        st.pyplot(fig)
        
        # Final summary
        st.markdown("<div class='system-card'>", unsafe_allow_html=True)
        st.markdown("### AI System Learning & Adaptation")
        st.markdown("**LSTM Hotspot Prediction:**")
        st.markdown("- Prediction accuracy improved from 82% to 87%")
        st.markdown("- New temporal patterns identified for weekday mornings")
        st.markdown("- Weather impact coefficients refined")
        
        st.markdown("**Dynamic Pricing Model:**")
        st.markdown("- Q-table updated based on 245 new ride transactions")
        st.markdown("- Price elasticity model refined for peak hours")
        st.markdown("- New optimal price points identified for different weather conditions")
        
        st.markdown("**Waiting Queue System:**")
        st.markdown("- Priority algorithms optimized based on wait time outcomes")
        st.markdown("- New balancing factors implemented for subscription vs. loyalty")
        st.markdown("- Queue processing speed improved by 15%")
        
        st.markdown("**Ride Sharing Algorithm:**")
        st.markdown("- Route compatibility scoring improved")
        st.markdown("- New fare split model validated with 33% average savings")
        st.markdown("- Rider matching preferences updated")
        st.markdown("</div>", unsafe_allow_html=True)

    # Auto-play functionality
    if st.session_state.auto_play and st.session_state.step < len(steps) - 1:
        time.sleep(11 - auto_play_speed)  # Inverse relationship with speed slider
        st.session_state.step += 1
        st.rerun()

# Run the app
if __name__ == "__main__":
    main()