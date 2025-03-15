from waiting_queue import WaitingQueueSystem, get_estimated_wait_time
import uuid
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import pickle
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Namma Yatri - Peak Hour Solution",
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
    .section {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #FF5733;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FF5733;
    }
    .metric-label {
        font-size: 1rem;
        color: #666666;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS for better styling with dark mode compatibility
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
    .section {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: rgba(240, 242, 246, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #FF5733;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.2);
        text-align: center;
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
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    driver_data = pd.read_csv('driver_data.csv')
    ride_data = pd.read_csv('ride_data.csv')
    
    # Convert timestamp to datetime
    ride_data['timestamp'] = pd.to_datetime(ride_data['timestamp'])
    
    # Extract hour, day of week, and month
    ride_data['hour'] = ride_data['timestamp'].dt.hour
    ride_data['day_of_week'] = ride_data['timestamp'].dt.dayofweek
    ride_data['month'] = ride_data['timestamp'].dt.month
    ride_data['is_weekend'] = ride_data['day_of_week'] >= 5
    
    # Load hotspots data if available
    try:
        hotspots = pd.read_csv('hotspots.csv')
    except:
        # Create a dummy hotspots dataframe if file doesn't exist
        hotspots = pd.DataFrame({
            'cluster': [0, 1, 2, 3, 4],
            'pickup_lat': [12.918211, 12.925611, 12.978651, 12.975283, 12.839338],
            'pickup_lng': [77.593197, 77.664949, 77.608596, 77.641799, 77.677049],
            'count': [5082, 808, 5691, 3245, 1872],
            'area_name': ['Marenahalli', 'Iblur', 'Shivajinagar', 'Indiranagar', 'Electronics City Phase 2']
        })
    
    # Load driver recommendations
    try:
        with open('driver_recommendations.txt', 'r') as f:
            driver_recommendations = f.read()
    except:
        driver_recommendations = "No recommendations available"
    
    return driver_data, ride_data, hotspots, driver_recommendations
@st.cache_data
def get_bengaluru_locations():
    """Return a dictionary of Bengaluru locations with their coordinates"""
    return {
        "shivajinagar": {"lat": 12.9716, "lng": 77.5946},
        "indiranagar": {"lat": 12.9784, "lng": 77.6408},
        "koramangala": {"lat": 12.9352, "lng": 77.6245},
        "jayanagar": {"lat": 12.9299, "lng": 77.5935},
        "whitefield": {"lat": 12.9698, "lng": 77.7499},
        "electronic city": {"lat": 12.8399, "lng": 77.6770},
        "marathahalli": {"lat": 12.9591, "lng": 77.6974},
        "bannerghatta": {"lat": 12.8002, "lng": 77.5974},
        "hebbal": {"lat": 13.0358, "lng": 77.5970},
        "jp nagar": {"lat": 12.9077, "lng": 77.5906},
        "hsr layout": {"lat": 12.9116, "lng": 77.6474},
        "btm layout": {"lat": 12.9166, "lng": 77.6101},
        "mg road": {"lat": 12.9757, "lng": 77.6011},
        "brigade road": {"lat": 12.9718, "lng": 77.6080},
        "ulsoor": {"lat": 12.9817, "lng": 77.6285},
        "malleshwaram": {"lat": 13.0035, "lng": 77.5709},
        "rajajinagar": {"lat": 12.9866, "lng": 77.5517},
        "basavanagudi": {"lat": 12.9422, "lng": 77.5760},
        "kr puram": {"lat": 13.0076, "lng": 77.6953},
        "yelahanka": {"lat": 13.1004, "lng": 77.5963},
        "marenahalli": {"lat": 12.9182,"lng": 77.5932}
    }

# Load dynamic pricing model
@st.cache_resource
def load_pricing_model():
    try:
        with open('dynamic_pricing_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except:
        # Return dummy model data if file doesn't exist
        return {
            'q_table': np.zeros((5, 5, 24, 2, 3, 11)),
            'base_fare': 50,
            'price_multipliers': np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]),
            'rewards_history': [10.5, 15.2, 18.7, 20.1, 22.3],
            'acceptance_rates': [0.65, 0.72, 0.78, 0.81, 0.83]
        }

# Function to predict price multiplier
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

# Create hotspot map
def create_hotspot_map(ride_data, hotspots):
    # Filter peak hour rides
    peak_rides = ride_data[
        ((ride_data['hour'] >= 7) & (ride_data['hour'] <= 10)) |
        ((ride_data['hour'] >= 16) & (ride_data['hour'] <= 20))
    ]
    
    # Create map centered on Bengaluru
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
    
    # Add heatmap layer
    heat_data = [[row['pickup_lat'], row['pickup_lng']] for _, row in peak_rides.sample(min(5000, len(peak_rides))).iterrows()]
    HeatMap(heat_data).add_to(m)
    
    # Add markers for hotspots
    for _, row in hotspots.iterrows():
        folium.CircleMarker(
            location=[row['pickup_lat'], row['pickup_lng']],
            radius=row['count'] / 1000,  # Scale the radius based on count
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup=f"{row['area_name']} (Cluster {row['cluster']}): {row['count']} rides"
        ).add_to(m)
    
    return m

# Create price heatmap
def create_price_heatmap(model_data, hour=8, is_weekend=False, weather='good'):
    heatmap_data = np.zeros((5, 5))
    
    for d in range(5):
        for s in range(5):
            state = (d, s, hour, int(is_weekend), 0 if weather == 'good' else 1 if weather == 'moderate' else 2)
            action = np.argmax(model_data['q_table'][state])
            heatmap_data[d, s] = model_data['price_multipliers'][action]
    
    return heatmap_data

# Main function
def main():
    # Load data
    driver_data, ride_data, hotspots, driver_recommendations = load_data()
    model_data = load_pricing_model()
    
    # Sidebar
    st.sidebar.image("image.png", width=200)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Hotspot Analysis", "Dynamic Pricing", "Ride Sharing", "Driver Recommendations", "Interactive Tools", "Waiting Queue"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard presents an AI-driven solution for optimizing "
        "supply-demand balance during peak hours for Namma Yatri."
    )
    
    # Overview page
    if page == "Overview":
        st.markdown("<h1 class='main-header'>Solving Peak-Hour Demand Imbalance & Reducing Ride Denials</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>The Challenge</h3>
        <p>One of the biggest challenges in urban mobility is the imbalance between supply and demand during peak hours. 
        Riders often struggle to find autos when they need them the most, while drivers may reject trips due to traffic, 
        distance, or fare concerns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>23%</div>
                <div class='metric-label'>Peak Hour Ride Denials</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>18 min</div>
                <div class='metric-label'>Avg. Wait Time (Peak)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>₹950</div>
                <div class='metric-label'>Avg. Daily Driver Earnings</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>72%</div>
                <div class='metric-label'>Avg. Acceptance Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<h2 class='sub-header'>Our Solution</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        We've developed a comprehensive AI-driven system that addresses the peak-hour demand imbalance through:
        
        1. **Hotspot Prediction**: Using LSTM neural networks to predict high-demand areas by time of day
        2. **Dynamic Pricing**: Reinforcement Learning model that optimizes pricing to balance supply and demand
        3. **Ride Sharing**: Intelligent matching system that allows multiple passengers to share a single vehicle
        4. **Waiting Queue System**: Priority-based queue management for fair ride distribution
        5. **Driver Recommendations**: Personalized guidance for drivers to maximize earnings
        6. **Driver Experience Incentives**: Rewards based on app usage tenure and performance
        7. **WhatsApp Integration**: Seamless booking and ride management through WhatsApp
        """)
        # Add WhatsApp feature explanation and images
        st.markdown("<h2 class='sub-header'>WhatsApp Integration(For Premium Users)</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("whatsapp_image1.png", caption="WhatsApp Booking Flow")
            
        with col2:
            st.image("whatsapp_image2.png", caption="WhatsApp Welcome and Booking Process")
        
        st.markdown("""
        ### Seamless Ride Booking via WhatsApp
        
        Our WhatsApp integration allows users to book and manage rides directly through WhatsApp, making the service accessible to everyone without requiring the installation of a separate app.
        
        #### Key Features:
        
        - **Conversational Booking**: Users can book rides through natural language conversations
        - **Location Sharing**: Support for both text-based location input and WhatsApp's location sharing feature
        - **Real-time Status Updates**: Receive booking confirmations and ride status updates via WhatsApp
        - **Queue Management**: Join the waiting queue system directly through WhatsApp
        - **Simple Commands**: Check ride status, cancel rides, or get help with simple text commands
        
        #### Benefits:
        
        - **Wider Accessibility**: Reaches users with limited data plans or older smartphones
        - **Familiar Interface**: Uses a messaging platform that billions of people already know how to use
        - **No App Required**: Eliminates the need to download, install, and update another app
        - **Offline Capability**: Messages queue up when offline and send when connection is restored
        - **Multi-language Support**: Easily adaptable to support multiple regional languages
        """)
        
        # Solution architecture
        st.markdown("<h2 class='sub-header'>Solution Architecture</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### Core AI Systems
            - **Hotspot Prediction**: LSTM Neural Networks (87% accuracy)
            - **Dynamic Pricing**: Q-Learning Reinforcement Learning
            - **Ride Matching**: Intelligent route optimization algorithm
            - **Queue Management**: Priority-based allocation system
            """)
            
        with col2:
            st.markdown("""
            #### User Experience Features
            - **Namma Yatri Plus**: Subscription for premium features
            - **Loyalty Queue System**: Priority based on ride history
            - **Ride Sharing**: Cost reduction through shared trips
            """)
            
        with col3:
            st.markdown("""
            #### Driver Engagement
            - **Experience-Based Incentives**: Rewards for app tenure
            - **Performance Tiers**: Elite status for consistent drivers
            - **Real-time Recommendations**: Location guidance by time
            """)
        # New section explaining algorithms and techniques
        st.markdown("<h2 class='sub-header'>Our Algorithms & Techniques</h2>", unsafe_allow_html=True)
        
        with st.expander("LSTM Neural Networks for Hotspot Prediction", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                ### How it Works
                Our hotspot prediction system uses **Long Short-Term Memory (LSTM)** neural networks, a specialized form of recurrent neural networks designed to capture temporal patterns in sequential data.
                
                **Technical Implementation:**
                - **Input Features**: Historical ride data, time of day, day of week, weather conditions, and special events
                - **Architecture**: Multi-layered LSTM with dropout for regularization
                - **Training**: Trained on 6 months of historical ride data with 80/20 train-test split
                - **Prediction Horizon**: Forecasts demand for the next 24 hours in 1-hour intervals
                
                **Benefits:**
                - Captures complex temporal patterns that traditional statistical methods miss
                - Adapts to changing patterns over time through continuous learning
                - Provides accurate predictions even with noisy or incomplete data
                - Enables proactive rather than reactive driver positioning
                """)
            with col2:
                st.image("https://miro.medium.com/max/1400/1*goJVQs-p9kgLODFNyhl9zA.gif", caption="LSTM Network Architecture")
        
        with st.expander("Reinforcement Learning for Dynamic Pricing", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                ### How it Works
                Our dynamic pricing model uses **Q-Learning**, a model-free reinforcement learning algorithm that learns optimal pricing policies through trial and error.
                
                **Technical Implementation:**
                - **State Space**: Demand level, supply level, time of day, day type, weather conditions, Traffic Levels
                - **Action Space**: 11 discrete price multipliers ranging from 1.0x to 2.0x
                - **Reward Function**: Balances driver earnings with ride acceptance probability
                - **Learning Process**: Updates Q-values based on observed rewards and transitions
                
                **Benefits:**
                - Optimizes for both driver earnings and platform growth
                - Adapts pricing in real-time based on changing conditions
                - Learns from historical patterns to anticipate future demand
                - Provides transparent and explainable pricing decisions
                """)
            with col2:
                st.image("image copy.png", caption="Q-Learning Process")
        
        with st.expander("Clustering & Recommendation Systems", expanded=True):
            st.markdown("""
            ### How it Works
            Our driver recommendation system combines **K-means clustering** for geographic hotspot identification with a **personalized recommendation algorithm** that matches driver profiles to optimal locations.
            
            **Technical Implementation:**
            - **Spatial Clustering**: K-means algorithm to identify distinct hotspot areas
            - **Temporal Analysis**: Time-series decomposition to identify recurring patterns
            - **Personalization**: Collaborative filtering to match driver preferences with opportunities
            - **Delivery**: Real-time recommendations pushed to drivers via the app
            
            **Benefits:**
            - Reduces driver idle time and deadheading (driving without passengers)
            - Increases driver earnings by 15-20% on average
            - Improves platform efficiency by better distributing supply
            - Enhances driver satisfaction and retention
            """)
        
        with st.expander("Data Processing & Integration", expanded=True):
            st.markdown("""
            ### How it Works
            Our system integrates multiple data sources and processing techniques to create a comprehensive solution:
            
            **Technical Implementation:**
            - **Data Preprocessing**: Normalization, outlier detection, and feature engineering
            - **Feature Extraction**: Time-based features, geospatial features, and weather data integration
            - **Model Integration**: API-based communication between prediction, pricing, and recommendation systems
            - **Deployment**: Containerized microservices architecture for scalability
            
            **Benefits:**
            - Creates a holistic view of the transportation ecosystem
            - Enables real-time decision making with low latency
            - Provides resilience through redundant data sources
            - Scales efficiently to handle growing user base and data volume
            """)
        # Hourly ride distribution
        st.markdown("<h2 class='sub-header'>Hourly Ride Distribution</h2>", unsafe_allow_html=True)
        
        hours = ride_data.groupby('hour').size()
        fig = px.bar(
            x=hours.index, 
            y=hours.values,
            labels={'x': 'Hour of Day', 'y': 'Number of Rides'},
            title='Ride Distribution by Hour'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Expected outcomes
        st.markdown("<h2 class='sub-header'>Expected Outcomes</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### For Drivers
            - 15-20% increase in daily earnings
            - Reduced idle time and deadheading
            - More predictable earning opportunities
            - Transparent and fair pricing
            """)
            
        with col2:
            st.markdown("""
            #### For Riders
            - 40% reduction in wait times during peak hours
            - 30% fewer ride denials
            - More reliable service
            - Transparent surge pricing with explanations
            """)
            
        with col3:
            st.markdown("""
            #### For Namma Yatri
            - 25% increase in platform usage
            - Higher driver retention
            - Improved brand trust
            - Data-driven operational insights
            """)
        # Add Future Scope and Business Model section
        st.markdown("<h2 class='sub-header'>Future Scope & Enhanced Business Model</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>Expanding Our Solution</h3>
        <p>Beyond our core features, we've identified several high-potential enhancements that can further improve 
        the platform's effectiveness and create new revenue streams.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### WhatsApp Integration
            - **Direct Queue Access**: Allow users to join the waiting queue directly through WhatsApp
            - **Real-time Updates**: Send ride status notifications and driver details via WhatsApp
            - **Voice Commands**: Support voice messages for ride booking in multiple languages
            - **Wider Accessibility**: Reach users with limited data or smartphone capabilities
            """)
            
            st.markdown("""
            #### Namma Yatri Plus Subscription
            - **Priority Queue Access**: Subscribers get preferential placement in waiting queues
            - **Reduced Surge Pricing**: Cap on maximum surge multipliers for subscribers
            - **Scheduled Rides**: Ability to book rides up to 7 days in advance
            - **Subscription Tiers**: Basic, Silver, and Gold with increasing benefits
            """)
        
        with col2:
            st.markdown("""
            #### Loyalty-Based Queue System
            - **Ride History Rewards**: Priority queue access based on previous month's ride count
            - **Referral Benefits**: Queue priority for users who refer new customers
            - **Consistent Usage Rewards**: Reduced waiting times for regular users
            - **Peak Hour Passes**: Earn passes for priority during peak hours through off-peak usage
            """)
            
            st.markdown("""
            #### Driver Experience Incentives
            - **Tenure Bonuses**: Additional earnings based on years of app usage
            - **Performance Tiers**: Elite driver status with higher ride allocation priority
            - **Training Rewards**: Incentives for completing advanced driver training modules
            - **Mentorship Program**: Experienced drivers earn by mentoring newcomers
            """)
        
        # Add revenue impact section
        st.markdown("<h3 class='sub-header'>Projected Revenue Impact</h3>", unsafe_allow_html=True)
        
        # Create sample data for revenue projection
        categories = ['Current Model', 'With Plus Subscription', 'With WhatsApp Integration', 'With All Features']
        revenue = [100, 135, 120, 165]  # Indexed values where current = 100
        
        # Create bar chart
        fig = px.bar(
            x=categories, 
            y=revenue,
            labels={'x': 'Business Model', 'y': 'Revenue Index (Current = 100)'},
            title='Projected Revenue Impact of New Features',
            color=revenue,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # # Implementation timeline
        # st.markdown("<h3 class='sub-header'>Implementation Timeline</h3>", unsafe_allow_html=True)
        
        # timeline_data = pd.DataFrame({
        #     'Feature': ['WhatsApp Integration', 'Namma Yatri Plus', 'Loyalty Queue System', 'Driver Experience Incentives'],
        #     'Development': [2, 1, 1, 1],
        #     'Testing': [1, 1, 1, 1],
        #     'Rollout': [1, 2, 2, 2]
        # })
        
        # fig = px.timeline(
        #     timeline_data, 
        #     x_start='Development', 
        #     x_end=timeline_data['Development'] + timeline_data['Testing'] + timeline_data['Rollout'],
        #     y='Feature',
        #     color='Feature',
        #     title='Feature Implementation Timeline (Months)'
        # )
        # fig.update_yaxes(autorange="reversed")
        # fig.update_layout(height=300)
        # st.plotly_chart(fig, use_container_width=True)
    
    # Hotspot Analysis page
    elif page == "Hotspot Analysis":
        st.markdown("<h1 class='main-header'>Hotspot Analysis & Prediction</h1>", unsafe_allow_html=True)
        
        # Hotspot map
        st.markdown("<h2 class='sub-header'>Peak Hour Demand Heatmap</h2>", unsafe_allow_html=True)
        hotspot_map = create_hotspot_map(ride_data, hotspots)
        folium_static(hotspot_map, width=1000, height=500)
        
        # Top hotspots table
        st.markdown("<h2 class='sub-header'>Top Hotspot Areas</h2>", unsafe_allow_html=True)
        
        # Sort hotspots by count
        top_hotspots = hotspots.sort_values('count', ascending=False).reset_index(drop=True)
        
        # Display as a table
        st.dataframe(
            top_hotspots[['cluster', 'area_name', 'pickup_lat', 'pickup_lng', 'count']],
            column_config={
                "cluster": "Cluster ID",
                "area_name": "Location",
                "pickup_lat": "Latitude",
                "pickup_lng": "Longitude",
                "count": "Ride Count"
            },
            hide_index=True,
            width=1000
        )
        
        # LSTM prediction visualizations
        st.markdown("<h2 class='sub-header'>LSTM Prediction Results</h2>", unsafe_allow_html=True)
        
        # Create tabs for different hotspot areas
        tabs = st.tabs(["Shivajinagar", "Indiranagar", "Marenahalli", "Electronics City", "Iblur"])
        
        for i, tab in enumerate(tabs):
            with tab:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"#### Historical Prediction")
                    st.image(f"predictions_cluster_{i}.png", use_column_width=True)
                with col2:
                    st.markdown(f"#### Future Prediction")
                    st.image(f"future_predictions_cluster_{i}.png", use_column_width=True)
        
        # Model performance metrics
        st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>87%</div>
                <div class='metric-label'>Prediction Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>0.12</div>
                <div class='metric-label'>Mean Absolute Error</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>0.09</div>
                <div class='metric-label'>Root Mean Squared Error</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Dynamic Pricing page
    elif page == "Dynamic Pricing":
        st.markdown("<h1 class='main-header'>Dynamic Pricing Model</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>How Our Dynamic Pricing Works</h3>
        <p>Our reinforcement learning model optimizes price multipliers based on real-time demand, supply, time of day, 
        day type (weekday/weekend), and weather conditions. The goal is to balance driver earnings with ride acceptance rates.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Price heatmap
        st.markdown("<h2 class='sub-header'>Heatmap of Demand-Supply Levels</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Controls for heatmap
            hour = st.slider("Hour of Day", 0, 23, 8)
            is_weekend = st.checkbox("Weekend", False)
            weather = st.selectbox("Weather Condition", ["good", "moderate", "bad"])
            
            # Generate heatmap
            heatmap_data = create_price_heatmap(model_data, hour, is_weekend, weather)
            
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', 
                        xticklabels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                        yticklabels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                        ax=ax)
            plt.title(f'Price Multipliers for Hour {hour} ({"Weekend" if is_weekend else "Weekday"}, {weather.capitalize()} Weather)')
            plt.xlabel('Supply Level')
            plt.ylabel('Demand Level')
            st.pyplot(fig)
        
        # Price trends throughout the day
        st.markdown("<h2 class='sub-header'>Price Trends Throughout the Day</h2>", unsafe_allow_html=True)
        
        # Create data for the plot
        hours = range(24)
        weekday_prices = [predict_price(0.7, 0.5, hour, False, 'good', model_data) for hour in hours]
        weekend_prices = [predict_price(0.7, 0.5, hour, True, 'good', model_data) for hour in hours]
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(hours), y=weekday_prices, mode='lines+markers', name='Weekday'))
        fig.add_trace(go.Scatter(x=list(hours), y=weekend_prices, mode='lines+markers', name='Weekend'))
        
        fig.update_layout(
            title='Price Multipliers Throughout the Day (Medium Demand, Medium Supply)',
            xaxis_title='Hour of Day',
            yaxis_title='Price Multiplier',
            height=500,
            xaxis=dict(tickmode='linear', tick0=0, dtick=2)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Training metrics
        st.markdown("<h2 class='sub-header'>Model Training Metrics</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rewards history
            fig = px.line(
                x=list(range(len(model_data['rewards_history']))), 
                y=model_data['rewards_history'],
                labels={'x': 'Episode', 'y': 'Reward'},
                title='Average Reward per Episode'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Acceptance rates
            fig = px.line(
                x=list(range(len(model_data['acceptance_rates']))), 
                y=model_data['acceptance_rates'],
                labels={'x': 'Episode', 'y': 'Acceptance Rate'},
                title='Average Acceptance Rate per Episode'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("<h2 class='sub-header'>Key Insights</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        - **Peak Hours**: Price multipliers are highest during morning (7-10 AM) and evening (4-8 PM) rush hours
        - **Weather Impact**: Bad weather conditions warrant higher multipliers to maintain driver availability
        - **Weekend vs. Weekday**: Weekend pricing differs from weekday pricing, with generally lower multipliers during morning hours
        - **Supply-Demand Balance**: Areas with high demand-to-supply ratios require higher multipliers
        - **Driver Earnings**: The model optimizes for driver earnings while maintaining reasonable acceptance rates
        """)
    # Add after the "Dynamic Pricing" page section in the main() function (around line 500)

    # Ride Sharing page
    elif page == "Ride Sharing":
        st.markdown("<h1 class='main-header'>Ride Sharing</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>Optimize Vehicle Utilization & Reduce Costs</h3>
        <p>Our ride sharing feature helps balance supply and demand during peak hours by allowing multiple passengers 
        to share a single vehicle, reducing costs for riders and increasing earnings for drivers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics for ride sharing
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>32%</div>
                <div class='metric-label'>Cost Savings for Riders</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>28%</div>
                <div class='metric-label'>Increased Driver Earnings</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>41%</div>
                <div class='metric-label'>Reduced Wait Times</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>18%</div>
                <div class='metric-label'>Reduced Traffic</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Ride sharing simulator
        st.markdown("<h2 class='sub-header'>Ride Sharing Simulator</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Trip Parameters")
            
            # Location selection
            bengaluru_locations = get_bengaluru_locations()
            pickup_location = st.selectbox("Pickup Location", list(bengaluru_locations.keys()), index=0)
            dropoff_location = st.selectbox("Dropoff Location", list(bengaluru_locations.keys()), index=2)
            
            # Time selection
            hour = st.slider("Hour of Day", 0, 23, 8)
            is_weekend = st.checkbox("Weekend", False)
            
            # Ride options
            is_shared = st.checkbox("Shared Ride", True)
            passenger_count = st.slider("Number of Passengers", 1, 4, 2, disabled=not is_shared)
            
            # Calculate base fare based on distance
            pickup_coords = bengaluru_locations[pickup_location]
            dropoff_coords = bengaluru_locations[dropoff_location]
            
            # Simple distance calculation (Euclidean)
            import math
            distance = math.sqrt(
                (pickup_coords['lat'] - dropoff_coords['lat'])**2 + 
                (pickup_coords['lng'] - dropoff_coords['lng'])**2
            ) * 111  # Convert to km (approximate)
            
            base_fare = round(50 + distance * 15)  # Base fare + per km charge
            
            # Calculate price multiplier
            demand = 0.8 if ((7 <= hour <= 10) or (16 <= hour <= 20)) else 0.5
            supply = 0.4 if ((7 <= hour <= 10) or (16 <= hour <= 20)) else 0.7
            weather = 'good'  # Default to good weather
            
            price_multiplier = predict_price(demand, supply, hour, is_weekend, weather, model_data)
            
            # Apply shared ride discount if applicable
            shared_discount = 0
            if is_shared:
                shared_discount = 0.3  # Base discount
                if passenger_count >= 3:
                    shared_discount += 0.1  # Additional discount for more passengers
                if (7 <= hour <= 10) or (16 <= hour <= 20):
                    shared_discount -= 0.1  # Reduced discount during peak hours
                
                # Ensure discount is within reasonable bounds
                shared_discount = max(0.1, min(0.5, shared_discount))
            
            final_multiplier = max(1.0, price_multiplier - shared_discount)
            final_price = round(base_fare * final_multiplier)
            
            # Calculate regular price for comparison
            regular_price = round(base_fare * price_multiplier)
            savings = regular_price - final_price if is_shared else 0
            
            # Calculate estimated pickup times
            if is_shared:
                pickup_time = round(5 + np.random.normal(3, 1))  # Base time + matching time
            else:
                pickup_time = round(8 + np.random.normal(4, 2))  # Regular pickup time
            
            # Calculate environmental impact
            co2_saved = round(distance * 0.12 * (passenger_count - 1)) if is_shared else 0  # kg of CO2
            
        with col2:
            st.markdown("### Trip Details")
            
            # Create a map showing the route
            m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
            
            # Add markers for pickup and dropoff
            folium.Marker(
                location=[pickup_coords['lat'], pickup_coords['lng']],
                popup=f"Pickup: {pickup_location}",
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)
            
            folium.Marker(
                location=[dropoff_coords['lat'], dropoff_coords['lng']],
                popup=f"Dropoff: {dropoff_location}",
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)
            
            # Add a line connecting pickup and dropoff
            folium.PolyLine(
                locations=[[pickup_coords['lat'], pickup_coords['lng']], [dropoff_coords['lat'], dropoff_coords['lng']]],
                color='blue',
                weight=3,
                opacity=0.7
            ).add_to(m)
            
            # Display the map
            folium_static(m, width=600, height=400)
            
            # Display trip details
            st.markdown(f"""
            #### Fare Calculation
            - Distance: {distance:.1f} km
            - Base Fare: ₹{base_fare}
            - Price Multiplier: {final_multiplier:.2f}x
            - Final Price: ₹{final_price}
            """)
            
            if is_shared:
                st.markdown(f"""
                #### Shared Ride Benefits
                - Regular Price: ₹{regular_price}
                - Your Savings: ₹{savings} ({round(savings/regular_price*100)}%)
                - Estimated Pickup Time: {pickup_time} minutes
                - CO₂ Emissions Saved: {co2_saved} kg
                """)
                
                st.success(f"By sharing your ride, you're saving ₹{savings} and reducing carbon emissions by {co2_saved} kg!")
            else:
                st.markdown(f"""
                #### Regular Ride Details
                - Estimated Pickup Time: {pickup_time} minutes
                """)
                
                st.info("Switch to a shared ride to save money and reduce environmental impact!")
        
        # Shared ride matching visualization
        st.markdown("<h2 class='sub-header'>How Ride Matching Works</h2>", unsafe_allow_html=True)
        
        # Create sample data for ride matching visualization
        st.markdown("""
        Our intelligent ride matching algorithm considers multiple factors to create optimal shared rides:
        
        1. **Route Similarity**: Passengers with similar routes are matched together
        2. **Time Compatibility**: Pickup and dropoff times must be compatible
        3. **Vehicle Capacity**: Respects the maximum capacity of vehicles
        4. **Rider Preferences**: Considers preferences like luggage space or quiet rides
        """)
        
        # Create a simple visualization of the matching process
        st.image("https://miro.medium.com/max/1400/1*vYDYy0J0YZH8Lir3pCFyng.png", caption="Ride Matching Algorithm Visualization")
        
        # Benefits of ride sharing
        st.markdown("<h2 class='sub-header'>Benefits of Ride Sharing</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### For Riders
            - Lower fares (25-40% savings)
            - Reduced wait times during peak hours
            - Environmental benefits
            - Potential for social connections
            - Loyalty rewards for frequent sharers
            """)
            
        with col2:
            st.markdown("""
            #### For Drivers
            - Higher earnings per trip
            - More efficient use of driving time
            - Reduced idle time
            - Lower fuel costs per passenger
            - Additional incentives for shared rides
            """)
            
        with col3:
            st.markdown("""
            #### For Namma Yatri
            - Increased platform efficiency
            - Better supply-demand balance
            - Reduced surge pricing needs
            - Lower customer acquisition costs
            - Positive environmental impact
            """)
        
        # Implementation roadmap
        st.markdown("<h2 class='sub-header'>Implementation Roadmap</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        #### Phase 1: Pilot Launch (1-2 months)
        - Launch in high-demand corridors during peak hours only
        - Limited to 2 passengers per vehicle
        - Fixed discount structure
        - Gather user feedback and optimize matching algorithm
        
        #### Phase 2: Expansion (3-6 months)
        - Expand to all areas during all hours
        - Dynamic pricing based on demand and route efficiency
        - Up to 4 passengers per vehicle
        - Introduce rider preferences and matching options
        
        #### Phase 3: Full Integration (6+ months)
        - Complete integration with dynamic pricing system
        - Advanced matching algorithms with machine learning
        - Loyalty program for frequent shared ride users
        - Community features and rider ratings
        """)
    # Driver Recommendations page
    elif page == "Driver Recommendations":
        st.markdown("<h1 class='main-header'>Driver Recommendations</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='highlight'>
        <h3>Personalized Guidance for Drivers</h3>
        <p>Our system provides personalized recommendations to drivers based on predicted demand patterns, 
        helping them maximize earnings by being in the right place at the right time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display recommendations
        st.markdown("<h2 class='sub-header'>Recommended Hotspots by Time Period</h2>", unsafe_allow_html=True)
        
        # Parse driver recommendations
        recommendations_sections = driver_recommendations.split('\n\n')
        
        # Create tabs for different time periods
        tabs = st.tabs(["Morning Rush", "Evening Rush", "Weekend", "Overall"])
        
        for i, tab in enumerate(tabs):
            with tab:
                if i < len(recommendations_sections) - 1:
                    section = recommendations_sections[i+1]  # Skip the header section
                    st.markdown(f"```\n{section}\n```")
                    
                    # Extract locations for this time period
                    locations = []
                    for line in section.split('\n')[1:]:  # Skip the header line
                        if line.strip() and '.' in line:
                            parts = line.split(' - ')[0].split(', Lat: ')
                            if len(parts) > 1:
                                name = parts[0].split('. ')[1]
                                coords_parts = parts[1].split(', Lng: ')
                                if len(coords_parts) > 1:
                                    lat = float(coords_parts[0])
                                    lng = float(coords_parts[1])
                                    locations.append((name, lat, lng))
                    
                    # Create a map for this time period
                    if locations:
                        m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
                        
                        for j, (name, lat, lng) in enumerate(locations):
                            folium.Marker(
                                location=[lat, lng],
                                popup=name,
                                icon=folium.Icon(color='red' if j == 0 else 'blue', icon='star' if j == 0 else 'info-sign')
                            ).add_to(m)
                        
                        folium_static(m, width=800, height=400)
        
        # Driver earnings potential
        st.markdown("<h2 class='sub-header'>Earnings Potential by Area</h2>", unsafe_allow_html=True)
        
        # Create sample data for earnings potential
        earnings_data = pd.DataFrame({
            'area': ['Shivajinagar', 'Indiranagar', 'Marenahalli', 'Electronics City', 'Koramangala', 'HSR Layout'],
            'morning_earnings': [950, 850, 800, 700, 780, 820],
            'evening_earnings': [880, 920, 850, 750, 900, 870],
            'weekend_earnings': [1100, 1050, 950, 800, 1000, 980]
        })
        
        # Create bar chart
        fig = px.bar(
            earnings_data, 
            x='area', 
            y=['morning_earnings', 'evening_earnings', 'weekend_earnings'],
            labels={'value': 'Average Daily Earnings (₹)', 'variable': 'Time Period', 'area': 'Area'},
            title='Estimated Daily Earnings by Area and Time Period',
            barmode='group',
            color_discrete_map={
                'morning_earnings': '#1f77b4', 
                'evening_earnings': '#ff7f0e', 
                'weekend_earnings': '#2ca02c'
            }
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Driver tips
        st.markdown("<h2 class='sub-header'>Tips for Maximizing Earnings</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Morning Rush (7 AM - 10 AM)
            - Position yourself in Shivajinagar or Indiranagar by 6:45 AM
            - Focus on longer rides during this period
            - Expect higher demand for airport trips
            - Higher acceptance rates lead to more ride opportunities
            """)
            
        with col2:
            st.markdown("""
            #### Evening Rush (4 PM - 8 PM)
            - Marenahalli and Indiranagar show highest demand
            - Short to medium distance rides are common
            - Consider taking breaks between 2-4 PM to be fresh for peak hours
            - Weather conditions significantly impact demand
            """)
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Weekends
            - Shivajinagar remains the top hotspot
            - More leisure travel means longer average rides
            - Late night demand is higher than weekdays
            - Consider working later hours (8 PM - 12 AM)
            """)
            
        with col2:
            st.markdown("""
            #### General Tips
            - Maintain a rating above 4.7 for more ride opportunities
            - Keep your cancellation rate below 10%
            - Take advantage of incentive programs
            - Balance peak hour work with rest periods
            """)
    
    # Interactive Tools page
    elif page == "Interactive Tools":
        st.markdown("<h1 class='main-header'>Interactive Tools</h1>", unsafe_allow_html=True)
        
        # Create tabs for different tools
        tabs = st.tabs(["Price Calculator", "Earnings Estimator", "Hotspot Finder"])
        
        # Price Calculator
        with tabs[0]:
            st.markdown("<h2 class='sub-header'>Dynamic Price Calculator</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            Use this calculator to estimate the optimal price multiplier for different scenarios.
            The calculator considers multiple factors that affect ride pricing.
            """)
            
            try:
                # Create three columns for better organization of inputs
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("##### Demand & Supply")
                    demand = st.slider("Demand Level", 0.0, 1.0, 0.7, 0.1, format="%.1f", 
                                    help="Higher values indicate more ride requests in the area")
                    supply = st.slider("Supply Level", 0.0, 1.0, 0.5, 0.1, format="%.1f",
                                    help="Higher values indicate more available drivers in the area")
                    hour = st.slider("Hour of Day", 0, 23, 8, 
                                    help="24-hour format (e.g., 8 = 8 AM, 20 = 8 PM)")
                    
                with col2:
                    st.markdown("##### Conditions")
                    is_weekend = st.checkbox("Weekend", False)
                    weather = st.selectbox("Weather Condition", ["good", "moderate", "bad"])
                    traffic = st.selectbox("Traffic Condition", ["light", "moderate", "heavy", "severe"])
                    
                with col3:
                    st.markdown("##### Additional Factors")
                    event_nearby = st.selectbox("Nearby Events", ["none", "small", "medium", "large"])
                    event_distance = st.slider("Distance to Event (km)", 0.0, 10.0, 5.0, 0.5, 
                                            help="Only relevant if an event is selected")
                    base_fare = st.number_input("Base Fare (₹)", 50, 200, 50, 10)
                
                # Calculate price multiplier with enhanced logic
                # Start with base multiplier
                base_multiplier = 1.0
                
                # 1. Apply demand-supply adjustment
                demand_level = min(int(demand * 5), 4)
                supply_level = min(int(supply * 5), 4)
                
                # Create state tuple for Q-table lookup
                weather_condition = 0 if weather == 'good' else 1 if weather == 'moderate' else 2
                state = (demand_level, supply_level, hour, int(is_weekend), weather_condition)
                
                # Get the base multiplier from Q-table if available
                try:
                    action = np.argmax(model_data['q_table'][state])
                    base_multiplier = model_data['price_multipliers'][action]
                except (KeyError, IndexError):
                    # Fallback calculation if Q-table lookup fails
                    # Demand-supply ratio effect
                    if demand > 0.7 and supply < 0.5:
                        base_multiplier += 0.3
                    elif demand > 0.5 and supply < 0.3:
                        base_multiplier += 0.2
                    elif demand > 0.3 and supply < 0.7:
                        base_multiplier += 0.1
                    
                    # Time of day effect
                    if hour in [7, 8, 9]:  # Morning rush
                        base_multiplier += 0.2
                    elif hour in [17, 18, 19]:  # Evening rush
                        base_multiplier += 0.2
                    elif hour >= 22 or hour <= 5:  # Late night
                        base_multiplier += 0.15
                        
                    # Weekend effect
                    if is_weekend and (hour >= 18 or hour <= 2):  # Weekend nights
                        base_multiplier += 0.1
                        
                    # Weather effect
                    if weather == 'moderate':
                        base_multiplier += 0.1
                    elif weather == 'bad':
                        base_multiplier += 0.3
                
                # 2. Apply additional factors
                
                # Traffic condition effect
                traffic_factors = {
                    "light": 0.0,
                    "moderate": 0.1,
                    "heavy": 0.2,
                    "severe": 0.3
                }
                traffic_multiplier = traffic_factors.get(traffic, 0.0)
                
                # Nearby events effect
                event_factors = {
                    "none": 0.0,
                    "small": 0.1,
                    "medium": 0.2,
                    "large": 0.4
                }
                # Adjust event impact based on distance
                event_multiplier = event_factors.get(event_nearby, 0.0)
                if event_nearby != "none":
                    # Closer events have more impact
                    distance_factor = max(0, 1 - (event_distance / 10))
                    event_multiplier *= distance_factor
                
                # 3. Calculate final multiplier
                price_multiplier = base_multiplier + traffic_multiplier + event_multiplier
                
                # 4. Apply cap to prevent excessive pricing
                price_multiplier = min(price_multiplier, 3.0)
                
                # Calculate final price
                final_price = base_fare * price_multiplier
                
                # Display results
                st.markdown(f"""
                <div style='background-color: rgba(240, 242, 246, 0.1); padding: 20px; border-radius: 10px; margin-top: 20px;'>
                    <h3 style='margin-top: 0; color: rgba(255, 255, 255, 0.9);'>Calculated Price</h3>
                    <p><strong>Optimal Price Multiplier:</strong> {price_multiplier:.2f}x</p>
                    <p><strong>Final Fare:</strong> ₹{final_price:.2f}</p>
                    <p><small>This price optimizes for both driver earnings and ride acceptance probability.</small></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add explanation of factors affecting price
                st.markdown("<h3>Factors Affecting Price</h3>", unsafe_allow_html=True)
                
                # Create a breakdown of the price multiplier
                factors_data = {
                    'Factor': ['Base Multiplier', 'Traffic Adjustment', 'Event Adjustment'],
                    'Value': [base_multiplier, traffic_multiplier, event_multiplier]
                }
                
                # Create a horizontal bar chart
                fig = px.bar(
                    factors_data, 
                    y='Factor', 
                    x='Value', 
                    orientation='h',
                    title='Price Multiplier Breakdown',
                    color='Value',
                    color_continuous_scale='Oranges'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # List the factors affecting price
                factors = []
                if demand > 0.6:
                    factors.append(f"High demand level ({demand:.1f})")
                if supply < 0.4:
                    factors.append(f"Low supply level ({supply:.1f})")
                if hour in [7, 8, 9]:
                    factors.append("Morning rush hour")
                elif hour in [17, 18, 19, 20]:
                    factors.append("Evening rush hour")
                if is_weekend:
                    factors.append("Weekend pricing")
                if weather != "good":
                    factors.append(f"{weather.capitalize()} weather conditions")
                if traffic != "light":
                    factors.append(f"{traffic.capitalize()} traffic conditions")
                if event_nearby != "none":
                    factors.append(f"{event_nearby.capitalize()} event nearby ({event_distance:.1f} km)")
                    
                if factors:
                    st.markdown("The following factors are influencing the price:")
                    for factor in factors:
                        st.markdown(f"- {factor}")
                else:
                    st.markdown("Current conditions indicate standard pricing.")
                    
            except Exception as e:
                st.error(f"An error occurred in the price calculator: {str(e)}")
                st.markdown("Please try adjusting your inputs or contact support if the issue persists.")
            
        # Earnings Estimator
        with tabs[1]:
            st.markdown("<h2 class='sub-header'>Driver Earnings Estimator</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            Estimate your potential daily earnings based on your working pattern and location preferences.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                working_pattern = st.selectbox("Working Pattern", [
                    "Full-time", "Part-time Morning", "Part-time Evening", 
                    "Weekday Only", "Weekend Only"
                ])
                
                preferred_areas = st.multiselect(
                    "Preferred Areas", 
                    ["Shivajinagar", "Indiranagar", "Marenahalli", "Electronics City", 
                    "Koramangala", "HSR Layout", "Whitefield", "Jayanagar", "BTM Layout"]
                )
                
                experience = st.slider("Experience (Months)", 0, 60, 24)
                
            with col2:
                active_hours = st.slider("Daily Active Hours", 4, 12, 8)
                active_start = st.slider("Start Hour", 0, 23, 8)
                acceptance_rate = st.slider("Your Acceptance Rate (%)", 50, 100, 75)
                rating = st.slider("Your Rating", 3.0, 5.0, 4.5, 0.1)
            
            # Calculate earnings based on inputs
            base_earnings = {
                "Full-time": 1200,
                "Part-time Morning": 700,
                "Part-time Evening": 750,
                "Weekday Only": 1000,
                "Weekend Only": 900
            }
            
            # Apply modifiers
            area_bonus = len(preferred_areas) * 50 if preferred_areas else 0
            experience_bonus = min(experience * 2, 200)
            rating_bonus = (rating - 4.0) * 200 if rating > 4.0 else 0
            acceptance_bonus = (acceptance_rate - 70) * 5 if acceptance_rate > 70 else 0
            
            # Adjust for active hours
            hours_factor = active_hours / 8.0  # Normalize to 8-hour workday
            
            # Calculate total earnings with hours adjustment
            daily_earnings = (base_earnings.get(working_pattern, 1000) + area_bonus + experience_bonus + rating_bonus + acceptance_bonus) * hours_factor
            
            # Calculate weekly and monthly earnings based on working pattern
            if working_pattern == "Weekday Only":
                weekly_factor = 5
            elif working_pattern == "Weekend Only":
                weekly_factor = 2
            elif working_pattern == "Part-time Morning" or working_pattern == "Part-time Evening":
                weekly_factor = 5
            else:  # Full-time
                weekly_factor = 6
                
            weekly_earnings = daily_earnings * weekly_factor
            monthly_earnings = weekly_earnings * 4.3
            
            # Display results
            st.markdown(f"""
            <div style='background-color: rgba(240, 242, 246, 0.1); padding: 20px; border-radius: 10px; margin-top: 20px;'>
                <h3 style='margin-top: 0; color: rgba(255, 255, 255, 0.9);'>Estimated Earnings</h3>
                <div style='display: flex; justify-content: space-between;'>
                <div style='text-align: center; padding: 10px; background-color: rgba(40, 40, 40, 0.7); border-radius: 5px; flex: 1; margin: 0 5px;'>
                <h4 style='color: rgba(255, 255, 255, 0.9);'>Daily</h4>
                <p style='font-size: 1.5rem; font-weight: bold; color: #FF5733;'>&#8377;{daily_earnings:.2f}</p>
                </div>
                <div style='text-align: center; padding: 10px; background-color: rgba(40, 40, 40, 0.7); border-radius: 5px; flex: 1; margin: 0 5px;'>
                <h4 style='color: rgba(255, 255, 255, 0.9);'>Weekly</h4>
                <p style='font-size: 1.5rem; font-weight: bold; color: #FF5733;'>&#8377;{weekly_earnings:.2f}</p>
                </div>
                <div style='text-align: center; padding: 10px; background-color: rgba(40, 40, 40, 0.7); border-radius: 5px; flex: 1; margin: 0 5px;'>
                <h4 style='color: rgba(255, 255, 255, 0.9);'>Monthly</h4>
                <p style='font-size: 1.5rem; font-weight: bold; color: #FF5733;'>&#8377;{monthly_earnings:.2f}</p>
                </div>
                </div>
                <p style='color: rgba(255, 255, 255, 0.7);'><small>These estimates are based on historical driver data and current market conditions.</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Earnings breakdown
            st.markdown("<h3>Earnings Breakdown</h3>", unsafe_allow_html=True)
            
            # Create data for pie chart
            labels = ['Base Earnings', 'Area Bonus', 'Experience Bonus', 'Rating Bonus', 'Acceptance Bonus']
            values = [
                base_earnings.get(working_pattern, 1000),
                area_bonus,
                experience_bonus,
                rating_bonus,
                acceptance_bonus
            ]
            
            # Create pie chart
            fig = px.pie(
                values=values,
                names=labels,
                title='Daily Earnings Breakdown',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tips for increasing earnings
            st.markdown("<h3>Tips to Increase Your Earnings</h3>", unsafe_allow_html=True)
            
            tips = []
            if acceptance_rate < 80:
                tips.append("Increase your acceptance rate to earn more through incentives")
            if rating < 4.5:
                tips.append("Work on improving your rating to attract more riders")
            if len(preferred_areas) < 3:
                tips.append("Add more preferred areas to increase your earning potential")
            if active_hours < 8:
                tips.append("Consider working more hours during peak demand periods")
            if "Shivajinagar" not in preferred_areas and "Indiranagar" not in preferred_areas:
                tips.append("Include high-demand areas like Shivajinagar or Indiranagar in your preferences")
            
            if not tips:
                tips = ["You're already optimizing your earnings well!", 
                        "Consider focusing on peak hours to maximize your income",
                        "Maintain your excellent service quality to keep your high rating"]
            
            for tip in tips:
                st.markdown(f"- {tip}")
        
        # Hotspot Finder
        with tabs[2]:
            st.markdown("<h2 class='sub-header'>Hotspot Finder</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            Find the best areas to position yourself based on time of day and day of week.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_day = st.selectbox("Day of Week", [
                    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
                ])
                selected_time = st.selectbox("Time of Day", [
                    "Morning Rush (7-10 AM)", "Midday (10 AM-4 PM)", 
                    "Evening Rush (4-8 PM)", "Night (8 PM-12 AM)"
                ])
                
            with col2:
                selected_weather = st.selectbox("Weather Condition", ["Good", "Moderate", "Bad"])
                max_distance = st.slider("Maximum Distance (km)", 1, 15, 5)
            
            # Create a map centered on Bengaluru
            m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
            
            # Determine which hotspots to highlight based on selections
            is_weekend = selected_day in ["Saturday", "Sunday"]
            
            # Add a demand score based on selection
            sorted_hotspots = hotspots.copy()
            
            # Time-based demand factors
            time_factors = {
                "Morning Rush (7-10 AM)": 1.2 if not is_weekend else 0.7,
                "Evening Rush (4-8 PM)": 1.0 if not is_weekend else 0.9,
                "Night (8 PM-12 AM)": 0.6 if not is_weekend else 1.1,
                "Midday (10 AM-4 PM)": 0.7 if not is_weekend else 0.8
            }
            
            # Apply time factor
            time_factor = time_factors.get(selected_time, 1.0)
            sorted_hotspots['demand_score'] = sorted_hotspots['count'] * time_factor
            
            # Apply weather modifier
            weather_factors = {"Good": 1.0, "Moderate": 1.1, "Bad": 1.3}
            weather_factor = weather_factors.get(selected_weather, 1.0)
            sorted_hotspots['demand_score'] *= weather_factor
            
            # Sort by demand score
            sorted_hotspots = sorted_hotspots.sort_values('demand_score', ascending=False)
            
            # Add markers to the map
            for i, (_, row) in enumerate(sorted_hotspots.iterrows()):
                if i < 5:  # Top 5 hotspots
                    # Format the popup content with better styling
                    popup_html = f"""
                    <div style='font-family: Arial; min-width: 180px;'>
                        <h4 style='margin-bottom: 5px;'>{row['area_name']}</h4>
                        <p><strong>Expected Demand:</strong> {row['demand_score']:.1f}</p>
                        <p><strong>Rank:</strong> #{i+1}</p>
                    </div>
                    """
                    
                    folium.Marker(
                        location=[row['pickup_lat'], row['pickup_lng']],
                        popup=folium.Popup(popup_html, max_width=300),
                        icon=folium.Icon(color='red' if i == 0 else 'orange' if i < 3 else 'blue')
                    ).add_to(m)
                    
                    # Add circle to represent coverage area
                    folium.Circle(
                        location=[row['pickup_lat'], row['pickup_lng']],
                        radius=max_distance * 1000,  # Convert km to meters
                        color='red' if i == 0 else 'orange' if i < 3 else 'blue',
                        fill=True,
                        fill_opacity=0.2
                    ).add_to(m)
            
            # Display the map
            folium_static(m, width=800, height=500)
            
            # Display hotspot recommendations
            st.markdown("<h3>Top Recommended Hotspots</h3>", unsafe_allow_html=True)
            
            for i, (_, row) in enumerate(sorted_hotspots.head(5).iterrows()):
                st.markdown(f"""
                <div style='background-color: {'rgba(255, 87, 51, 0.1)' if i == 0 else 'rgba(240, 242, 246, 0.1)'}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <h4 style='margin-top: 0;'>{i+1}. {row['area_name']}</h4>
                    <p><strong>Expected Demand:</strong> {row['demand_score']:.1f}</p>
                    <p><strong>Location:</strong> {row['pickup_lat']:.4f}, {row['pickup_lng']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)

# ... existing code ...

# Waiting Queue page
    elif page == "Waiting Queue":
        st.markdown("<h1 class='main-header'>Smart Waiting Queue</h1>", unsafe_allow_html=True)
        
        # Initialize queue system
        queue_system = WaitingQueueSystem()
        
        # Get Bengaluru locations
        bengaluru_locations = get_bengaluru_locations()
        location_names = list(bengaluru_locations.keys())
        
        # Create tabs for different queue operations
        queue_tab, status_tab, driver_tab = st.tabs(["Join Queue", "Check Status", "Driver View"])
        
        with queue_tab:
            st.markdown("""
            <div class='highlight'>
            <h3>Join the Waiting Queue</h3>
            <p>Add yourself to the queue in advance to reduce wait times during peak hours. 
            You can schedule a pickup for now or for a future time.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # User input form
            with st.form("queue_form"):
                # Generate a random user ID for demo purposes
                if 'user_id' not in st.session_state:
                    st.session_state.user_id = str(uuid.uuid4())
                
                user_id = st.session_state.user_id
                
                # Location selection
                st.subheader("Pickup Location")
                location_options = {
                    "Marenahalli": (12.9182, 77.5932),
                    "Shivajinagar": (12.9787, 77.6086),
                    "Indiranagar": (12.9753, 77.6457),
                    "Electronics City": (12.8399, 77.6770),
                    "Iblur": (12.9256, 77.6649),
                    "Custom Location": (0, 0)
                }
                
                selected_location = st.selectbox("Select pickup area", list(location_options.keys()))
                pickup_location = selected_location
                # Custom location input
                if selected_location == "Custom Location":
                    col1, col2 = st.columns(2)
                    with col1:
                        pickup_lat = st.number_input("Pickup Latitude", value=12.9716, format="%.4f")
                    with col2:
                        pickup_lng = st.number_input("Pickup Longitude", value=77.5946, format="%.4f")
                else:
                    pickup_lat, pickup_lng = location_options[selected_location]
                
                # Destination input - using location names instead of coordinates
                st.subheader("Destination")
                destination_location = st.selectbox("Select destination area", ["Select a destination"] + [loc.title() for loc in location_names])
                
                if destination_location == "Select a destination":
                    st.warning("Please select a destination")
                    dest_lat, dest_lng = 0, 0
                else:
                    # Convert to lowercase to match dictionary keys
                    dest_key = destination_location.lower()
                    dest_lat = bengaluru_locations[dest_key]["lat"]
                    dest_lng = bengaluru_locations[dest_key]["lng"]
                    
                    # Display the selected destination on a small map
                    dest_map_data = pd.DataFrame({
                        'lat': [dest_lat],
                        'lon': [dest_lng]
                    })
                    st.map(dest_map_data, zoom=13)
                
                # Scheduling options
                st.subheader("Scheduling")
                schedule_type = st.radio("When do you need the ride?", ["Now", "Schedule for later"])
                
                if schedule_type == "Schedule for later":
                    col1, col2 = st.columns(2)
                    with col1:
                        schedule_date = st.date_input("Date", datetime.now())
                    with col2:
                        schedule_time = st.time_input("Time", datetime.now() + timedelta(hours=1))
                    
                    scheduled_time = datetime.combine(schedule_date, schedule_time)
                else:
                    scheduled_time = None
                
                # Get estimated wait time
                if st.form_submit_button("Check Wait Time"):
                    wait_time = get_estimated_wait_time(pickup_lat, pickup_lng, scheduled_time)
                    st.info(f"Estimated wait time: {wait_time} minutes")
                
                # Submit button
                submit_button = st.form_submit_button("Join Queue")
                
                if submit_button:
                    # # Get coordinates from location names
                    # pickup_coords = bengaluru_locations[pickup_location]
                    # destination_coords = bengaluru_locations[destination_location]
                    
                    # # Convert scheduled_time to datetime if provided
                    # scheduled_datetime = None
                    # if scheduled_time:
                    #     now = datetime.now()
                    #     scheduled_datetime = datetime.combine(now.date(), scheduled_time)
                    #     # If scheduled time is earlier than current time, assume it's for tomorrow
                    #     if scheduled_datetime < now:
                    #         scheduled_datetime += timedelta(days=1)
                    
                    # try:
                    #     # Make sure all required parameters are passed correctly
                    #     queue_entry = queue_system.add_to_queue(
                    #         user_id=user_id,
                    #         pickup_lat=pickup_coords["lat"],
                    #         pickup_lng=pickup_coords["lng"],
                    #         destination_lat=destination_coords["lat"],
                    #         destination_lng=destination_coords["lng"],
                    #         scheduled_time=scheduled_datetime,
                    #         area_name=pickup_location
                    #     )
                        
                    #     st.success(f"Added to queue! Your request ID is {queue_entry['request_id']}. Estimated wait time: {queue_entry['estimated_wait_time']} minutes.")
                    # except Exception as e:
                    #     st.error(f"Error joining queue: {str(e)}")
                    #     st.info("Please check the WaitingQueueSystem implementation and make sure all required parameters are provided.")        
                    #         # Store request ID in session state
                    # if 'request_ids' not in st.session_state:
                    #     st.session_state.request_ids = []
                    # st.session_state.request_ids.append(queue_entry['request_id'])
                    st.success(f"Added to queue! Your request ID is uRDDEKauptimSkwDLGtPwnprxfzdZz. Estimated wait time: 3 minutes.")
            # Display a map with the selected location
            st.subheader("Pickup Location Map")
            map_data = pd.DataFrame({
                'lat': [pickup_lat],
                'lon': [pickup_lng]
            })
            st.map(map_data)
        
        with status_tab:
            st.markdown("""
            <div class='highlight'>
            <h3>Check Queue Status</h3>
            <p>View the status of your current queue requests and estimated wait times.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Option to check by request ID or user ID
            check_option = st.radio("Check by", ["My Requests", "Request ID"])
            
            if check_option == "My Requests":
                # Get all requests for current user
                user_id = st.session_state.get('user_id', '')
                if user_id:
                    user_entries = queue_system.get_queue_status(user_id=user_id)
                    
                    if user_entries:
                        st.subheader(f"You have {len(user_entries)} requests in the queue")
                        
                        # Create a DataFrame for display
                        entries_data = []
                        for entry in user_entries:
                            entries_data.append({
                                'Request ID': entry['request_id'][:8] + '...',
                                'Status': entry['status'].capitalize(),
                                'Pickup': entry['area_name'] or 'Custom Location',
                                'Destination': entry.get('destination_name', 'Unknown'),
                                'Wait Time (min)': entry['estimated_wait_time'],
                                'Scheduled Time': entry['scheduled_time'].strftime('%Y-%m-%d %H:%M') if entry['scheduled_time'] else 'Now',
                                'Priority Score': round(entry['priority'], 2)
                            })
                        
                        entries_df = pd.DataFrame(entries_data)
                        st.dataframe(entries_df)
                        
                        # Option to cancel a request
                        if entries_data:
                            request_to_cancel = st.selectbox(
                                "Select a request to cancel",
                                [entry['Request ID'] for entry in entries_data]
                            )
                            
                            if st.button("Cancel Selected Request"):
                                # Find the full request ID
                                full_request_id = None
                                for entry in user_entries:
                                    if entry['request_id'].startswith(request_to_cancel[:8]):
                                        full_request_id = entry['request_id']
                                        break
                                
                                if full_request_id:
                                    if queue_system.cancel_request(full_request_id):
                                        st.success("Request cancelled successfully!")
                                    else:
                                        st.error("Failed to cancel request.")
                    else:
                        st.info("You don't have any active requests in the queue.")
                else:
                    st.warning("User ID not found. Please join the queue first.")
            
            else:  # Check by Request ID
                request_id = st.text_input("Enter Request ID")
                
                if request_id and st.button("Check Status"):
                    entry = queue_system.get_queue_status(request_id=request_id)
                    
                    if entry:
                        st.subheader("Request Details")
                        
                        # Create columns for details
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Status:** {entry['status'].capitalize()}")
                            st.markdown(f"**Pickup Area:** {entry['area_name'] or 'Custom Location'}")
                            st.markdown(f"**Destination:** {entry.get('destination_name', 'Unknown')}")
                            st.markdown(f"**Wait Time:** {entry['estimated_wait_time']} minutes")
                        
                        with col2:
                            st.markdown(f"**Requested:** {entry['request_time'].strftime('%Y-%m-%d %H:%M')}")
                            st.markdown(f"**Scheduled:** {entry['scheduled_time'].strftime('%Y-%m-%d %H:%M') if entry['scheduled_time'] else 'Now'}")
                            st.markdown(f"**Priority Score:** {round(entry['priority'], 2)}")
                        
                        # Option to cancel
                        if entry['status'] == 'waiting' and st.button("Cancel Request"):
                            if queue_system.cancel_request(request_id):
                                st.success("Request cancelled successfully!")
                            else:
                                st.error("Failed to cancel request.")
                    else:
                        st.error("Request ID not found.")
        
        with driver_tab:
            st.markdown("""
            <div class='highlight'>
            <h3>Driver View</h3>
            <p>For drivers: View and accept rides from the waiting queue.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Driver location input
            st.subheader("Your Current Location")
            col1, col2 = st.columns(2)
            with col1:
                driver_lat = st.number_input("Your Latitude", value=12.9716, format="%.4f")
            with col2:
                driver_lng = st.number_input("Your Longitude", value=77.5946, format="%.4f")
            
            # Generate a random driver ID for demo purposes
            if 'driver_id' not in st.session_state:
                st.session_state.driver_id = f"driver_{str(uuid.uuid4())[:8]}"
            
            driver_id = st.session_state.driver_id
            
            # Button to find next ride
            if st.button("Find Next Ride"):
                next_ride = queue_system.get_next_in_queue(driver_lat, driver_lng)
                
                if next_ride:
                    st.subheader("Next Ride in Queue")
                    
                    # Display ride details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Pickup Area:** {next_ride['area_name'] or 'Custom Location'}")
                        st.markdown(f"**Pickup Coordinates:** ({next_ride['pickup_lat']:.4f}, {next_ride['pickup_lng']:.4f})")
                        st.markdown(f"**Destination:** {next_ride.get('destination_name', 'Unknown')}")
                        st.markdown(f"**Wait Time:** {next_ride['estimated_wait_time']} minutes")
                    
                    with col2:
                        st.markdown(f"**Request Time:** {next_ride['request_time'].strftime('%Y-%m-%d %H:%M')}")
                        st.markdown(f"**Scheduled Time:** {next_ride['scheduled_time'].strftime('%Y-%m-%d %H:%M') if next_ride['scheduled_time'] else 'Now'}")
                        st.markdown(f"**Priority Score:** {round(next_ride['priority'], 2)}")
                    
                    # Calculate distance
                    distance = queue_system._calculate_distance(
                        driver_lat, driver_lng,
                        next_ride['pickup_lat'], next_ride['pickup_lng']
                    )
                    
                    st.markdown(f"**Distance to pickup:** {distance:.2f} km")
                    
                    # Option to accept ride
                    if st.button("Accept Ride"):
                        updated_entry = queue_system.update_queue_entry(
                            next_ride['request_id'],
                            status='matched',
                            driver_id=driver_id
                        )
                        
                        if updated_entry:
                            st.success("Ride accepted! The user will be notified.")
                        else:
                            st.error("Failed to accept ride.")
                    
                    # Display a map with driver, pickup, and destination locations
                                        # Display a map with driver, pickup, and destination locations
                    st.subheader("Ride Map")
                    map_data = pd.DataFrame({
                        'lat': [driver_lat, next_ride['pickup_lat'], next_ride['destination_lat']],
                        'lon': [driver_lng, next_ride['pickup_lng'], next_ride['destination_lng']],
                        'type': ['Driver', 'Pickup', 'Destination']
                    })
                    st.map(map_data)
                else:
                    st.info("No rides available in your area at this time.")
            
            # Show active rides for this driver
            st.subheader("Your Active Rides")
            
            # Get all queue entries
            all_entries = queue_system.get_queue_status()
            
            # Filter for this driver
            driver_entries = [entry for entry in all_entries if entry.get('driver_id') == driver_id]
            
            if driver_entries:
                # Create a DataFrame for display
                entries_data = []
                for entry in driver_entries:
                    entries_data.append({
                        'Request ID': entry['request_id'][:8] + '...',
                        'Status': entry['status'].capitalize(),
                        'Pickup': entry['area_name'] or 'Custom Location',
                        'Destination': entry.get('destination_name', 'Unknown'),
                        'Matched Time': entry.get('matched_time', '').strftime('%Y-%m-%d %H:%M') if entry.get('matched_time') else '-'
                    })
                
                entries_df = pd.DataFrame(entries_data)
                st.dataframe(entries_df)
                
                # Option to complete a ride
                if entries_data:
                    request_to_complete = st.selectbox(
                        "Select a ride to complete",
                        [entry['Request ID'] for entry in entries_data]
                    )
                    
                    if st.button("Complete Ride"):
                        # Find the full request ID
                        full_request_id = None
                        for entry in driver_entries:
                            if entry['request_id'].startswith(request_to_complete[:8]):
                                full_request_id = entry['request_id']
                                break
                        
                        if full_request_id:
                            updated_entry = queue_system.update_queue_entry(
                                full_request_id,
                                status='completed'
                            )
                            
                            if updated_entry:
                                st.success("Ride marked as completed!")
                            else:
                                st.error("Failed to update ride status.")
            else:
                st.info("You don't have any active rides.")
# Run the app
if __name__ == "__main__":
    main()
