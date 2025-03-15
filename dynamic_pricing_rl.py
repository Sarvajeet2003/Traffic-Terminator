import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import os

class DynamicPricingRL:
    def __init__(self, base_fare=50, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        """
        Initialize the RL model for dynamic pricing
        
        Parameters:
        -----------
        base_fare : float
            Base fare for rides in INR
        learning_rate : float
            Learning rate for Q-learning (alpha)
        discount_factor : float
            Discount factor for future rewards (gamma)
        exploration_rate : float
            Probability of exploring random actions (epsilon)
        """
        self.base_fare = base_fare
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Define state space dimensions
        self.demand_levels = 5  # Very Low, Low, Medium, High, Very High
        self.supply_levels = 5  # Very Low, Low, Medium, High, Very High
        self.time_slots = 24    # Hours of the day
        self.day_types = 2      # Weekday or Weekend
        self.weather_conditions = 3  # Good, Moderate, Bad
        
        # Define action space (multiplier for base fare)
        self.price_multipliers = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        
        # Initialize Q-table
        self.q_table = np.zeros((
            self.demand_levels,
            self.supply_levels,
            self.time_slots,
            self.day_types,
            self.weather_conditions,
            len(self.price_multipliers)
        ))
        
        # Metrics tracking
        self.rewards_history = []
        self.acceptance_rates = []
        
    def get_state(self, demand, supply, hour, is_weekend, weather):
        """
        Convert continuous values to discrete state representation
        
        Parameters:
        -----------
        demand : float
            Current demand level (number of ride requests)
        supply : float
            Current supply level (number of available drivers)
        hour : int
            Current hour (0-23)
        is_weekend : bool
            Whether it's a weekend (True) or weekday (False)
        weather : str
            Weather condition ('good', 'moderate', 'bad')
            
        Returns:
        --------
        tuple
            Discrete state representation (demand_level, supply_level, hour, day_type, weather_condition)
        """
        # Discretize demand (assuming demand is normalized between 0 and 1)
        demand_level = min(int(demand * self.demand_levels), self.demand_levels - 1)
        
        # Discretize supply (assuming supply is normalized between 0 and 1)
        supply_level = min(int(supply * self.supply_levels), self.supply_levels - 1)
        
        # Time slot is already discrete (0-23)
        time_slot = hour
        
        # Day type (0 for weekday, 1 for weekend)
        day_type = 1 if is_weekend else 0
        
        # Weather condition
        if weather == 'good':
            weather_condition = 0
        elif weather == 'moderate':
            weather_condition = 1
        else:  # bad
            weather_condition = 2
            
        return (demand_level, supply_level, time_slot, day_type, weather_condition)
    
    def choose_action(self, state):
        """
        Choose price multiplier based on current state using epsilon-greedy policy
        
        Parameters:
        -----------
        state : tuple
            Current state (demand_level, supply_level, time_slot, day_type, weather_condition)
            
        Returns:
        --------
        int
            Index of the chosen price multiplier
        """
        # Exploration: choose a random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, len(self.price_multipliers))
        
        # Exploitation: choose the best action from Q-table
        return np.argmax(self.q_table[state])
    
    def calculate_reward(self, demand, supply, price_multiplier, acceptance_rate):
        """
        Calculate reward based on the action taken
        
        Parameters:
        -----------
        demand : float
            Current demand level
        supply : float
            Current supply level
        price_multiplier : float
            Applied price multiplier
        acceptance_rate : float
            Driver acceptance rate for the given price
            
        Returns:
        --------
        float
            Calculated reward
        """
        # Calculate demand-supply ratio
        if supply > 0:
            ds_ratio = demand / supply
        else:
            ds_ratio = 10.0  # High value when no supply
        
        # Calculate revenue
        revenue = self.base_fare * price_multiplier * acceptance_rate * min(demand, supply)
        
        # Penalty for unmet demand
        unmet_demand_penalty = max(0, demand - supply * acceptance_rate) * 10
        
        # Penalty for excess supply (idle drivers)
        excess_supply_penalty = max(0, supply - demand / acceptance_rate) * 5
        
        # Calculate final reward
        reward = revenue - unmet_demand_penalty - excess_supply_penalty
        
        return reward
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-table using Q-learning algorithm
        
        Parameters:
        -----------
        state : tuple
            Current state
        action : int
            Chosen action (index of price multiplier)
        reward : float
            Received reward
        next_state : tuple
            Next state
        """
        # Get current Q-value
        current_q = self.q_table[state][action]
        
        # Get maximum Q-value for next state
        max_next_q = np.max(self.q_table[next_state])
        
        # Calculate new Q-value
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[state][action] = new_q
        
    def predict_price(self, demand, supply, hour, is_weekend, weather):
        """
        Predict optimal price multiplier for given conditions
        
        Parameters:
        -----------
        demand : float
            Current demand level (normalized between 0 and 1)
        supply : float
            Current supply level (normalized between 0 and 1)
        hour : int
            Current hour (0-23)
        is_weekend : bool
            Whether it's a weekend (True) or weekday (False)
        weather : str
            Weather condition ('good', 'moderate', 'bad')
            
        Returns:
        --------
        float
            Optimal price multiplier
        """
        # Get state representation
        state = self.get_state(demand, supply, hour, is_weekend, weather)
        
        # Choose best action (no exploration during prediction)
        action = np.argmax(self.q_table[state])
        
        # Return corresponding price multiplier
        return self.price_multipliers[action]
    
    def train(self, data, episodes=1000):
        """
        Train the RL model using historical data
        
        Parameters:
        -----------
        data : DataFrame
            Historical data with columns: 'demand', 'supply', 'hour', 'is_weekend', 'weather', 'acceptance_rate'
        episodes : int
            Number of training episodes
        """
        print("Training dynamic pricing model...")
        
        for episode in range(episodes):
            total_reward = 0
            avg_acceptance = 0
            
            # Shuffle data for each episode
            data_sample = data.sample(frac=1).reset_index(drop=True)
            
            for i, row in data_sample.iterrows():
                # Get current state
                state = self.get_state(
                    row['demand'], 
                    row['supply'], 
                    row['hour'], 
                    row['is_weekend'], 
                    row['weather']
                )
                
                # Choose action
                action = self.choose_action(state)
                price_multiplier = self.price_multipliers[action]
                
                # Simulate driver acceptance rate based on price multiplier
                # Higher price = higher driver acceptance, but with diminishing returns
                base_acceptance = row['acceptance_rate']
                simulated_acceptance = min(1.0, base_acceptance * (1 + 0.2 * (price_multiplier - 1)))
                
                # Calculate reward
                reward = self.calculate_reward(
                    row['demand'], 
                    row['supply'], 
                    price_multiplier, 
                    simulated_acceptance
                )
                
                # Get next state (using next row or wrapping around)
                next_idx = (i + 1) % len(data_sample)
                next_row = data_sample.iloc[next_idx]
                
                next_state = self.get_state(
                    next_row['demand'], 
                    next_row['supply'], 
                    next_row['hour'], 
                    next_row['is_weekend'], 
                    next_row['weather']
                )
                
                # Update Q-table
                self.update_q_table(state, action, reward, next_state)
                
                total_reward += reward
                avg_acceptance += simulated_acceptance
            
            # Decay exploration rate
            self.exploration_rate = max(0.01, self.exploration_rate * 0.995)
            
            # Track metrics
            self.rewards_history.append(total_reward / len(data_sample))
            self.acceptance_rates.append(avg_acceptance / len(data_sample))
            
            # Print progress
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes}, Avg Reward: {self.rewards_history[-1]:.2f}, Acceptance Rate: {self.acceptance_rates[-1]:.2f}")
        
        print("Training completed!")
    
    def save_model(self, filename='dynamic_pricing_model.pkl'):
        """Save the trained model to a file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'base_fare': self.base_fare,
                'price_multipliers': self.price_multipliers,
                'rewards_history': self.rewards_history,
                'acceptance_rates': self.acceptance_rates
            }, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='dynamic_pricing_model.pkl'):
        """Load a trained model from a file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.base_fare = data['base_fare']
            self.price_multipliers = data['price_multipliers']
            self.rewards_history = data['rewards_history']
            self.acceptance_rates = data['acceptance_rates']
        print(f"Model loaded from {filename}")
    
    def plot_training_metrics(self):
        """Plot training metrics"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards_history)
        plt.title('Average Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.acceptance_rates)
        plt.title('Average Acceptance Rate per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Acceptance Rate')
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
    
    def generate_price_heatmap(self):
        """Generate heatmap of price multipliers for different demand-supply scenarios"""
        # Create a heatmap for a specific time and condition
        hour = 8  # Morning rush hour
        is_weekend = False
        weather = 'good'
        
        heatmap_data = np.zeros((self.demand_levels, self.supply_levels))
        
        for d in range(self.demand_levels):
            for s in range(self.supply_levels):
                state = (d, s, hour, int(is_weekend), 0)  # 0 for good weather
                action = np.argmax(self.q_table[state])
                heatmap_data[d, s] = self.price_multipliers[action]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', 
                    xticklabels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                    yticklabels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        plt.title(f'Price Multipliers for Hour {hour} ({"Weekend" if is_weekend else "Weekday"}, {weather.capitalize()} Weather)')
        plt.xlabel('Supply Level')
        plt.ylabel('Demand Level')
        plt.savefig('price_heatmap.png')
        plt.close()

# Function to prepare synthetic training data
def prepare_synthetic_data(ride_data, hotspots, n_samples=5000):
    """
    Prepare synthetic data for training the RL model
    
    Parameters:
    -----------
    ride_data : DataFrame
        Historical ride data
    hotspots : DataFrame
        Hotspot information
    n_samples : int
        Number of synthetic samples to generate
        
    Returns:
    --------
    DataFrame
        Synthetic data for RL training
    """
    # Extract time patterns from historical data
    ride_data['hour'] = ride_data['timestamp'].dt.hour
    ride_data['is_weekend'] = ride_data['timestamp'].dt.dayofweek >= 5
    
    # Calculate hourly demand patterns
    hourly_demand = ride_data.groupby(['hour', 'is_weekend']).size().reset_index(name='count')
    hourly_demand['demand'] = hourly_demand['count'] / hourly_demand['count'].max()
    
    # Create synthetic data
    synthetic_data = []
    
    # Weather probabilities
    weather_options = ['good', 'moderate', 'bad']
    weather_probs = [0.7, 0.2, 0.1]
    
    for _ in range(n_samples):
        # Randomly select hour and day type
        row = hourly_demand.sample(1).iloc[0]
        hour = row['hour']
        is_weekend = row['is_weekend']
        
        # Base demand from historical patterns
        base_demand = row['demand']
        
        # Add random variation
        demand = min(1.0, max(0.1, base_demand * np.random.normal(1, 0.2)))
        
        # Supply is correlated with demand but with some randomness
        # Lower supply during peak hours to simulate real-world conditions
        if 7 <= hour <= 10 or 16 <= hour <= 20:  # Peak hours
            supply = demand * np.random.normal(0.8, 0.2)  # Less supply during peak
        else:
            supply = demand * np.random.normal(1.2, 0.2)  # More supply during off-peak
        
        supply = min(1.0, max(0.1, max(0.1, supply)))
        
        # Random weather
        weather = np.random.choice(weather_options, p=weather_probs)
        
        # Base acceptance rate (higher during non-peak hours, lower during peak)
        if 7 <= hour <= 10 or 16 <= hour <= 20:  # Peak hours
            base_acceptance = np.random.uniform(0.5, 0.8)  # Lower acceptance during peak
        else:
            base_acceptance = np.random.uniform(0.7, 0.95)  # Higher acceptance during off-peak
        
        # Adjust acceptance based on weather
        if weather == 'moderate':
            base_acceptance *= 0.9
        elif weather == 'bad':
            base_acceptance *= 0.8
        
        synthetic_data.append({
            'demand': demand,
            'supply': supply,
            'hour': hour,
            'is_weekend': is_weekend,
            'weather': weather,
            'acceptance_rate': base_acceptance
        })
    
    return pd.DataFrame(synthetic_data)

# Function to visualize pricing recommendations
def visualize_pricing_recommendations(model, ride_data, hotspots):
    """
    Visualize pricing recommendations for different scenarios
    
    Parameters:
    -----------
    model : DynamicPricingRL
        Trained RL model
    ride_data : DataFrame
        Historical ride data
    hotspots : DataFrame
        Hotspot information
    """
    # Create a figure for pricing recommendations
    plt.figure(figsize=(15, 10))
    
    # 1. Price multipliers throughout the day (weekday)
    plt.subplot(2, 2, 1)
    hours = range(24)
    weekday_prices = [model.predict_price(0.7, 0.5, hour, False, 'good') for hour in hours]
    weekend_prices = [model.predict_price(0.7, 0.5, hour, True, 'good') for hour in hours]
    
    plt.plot(hours, weekday_prices, 'b-', label='Weekday')
    plt.plot(hours, weekend_prices, 'r-', label='Weekend')
    plt.title('Price Multipliers Throughout the Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Price Multiplier')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 2. Price multipliers for different demand levels (peak hour, weekday)
    plt.subplot(2, 2, 2)
    demand_levels = np.linspace(0, 1, 11)
    peak_hour_prices = [model.predict_price(demand, 0.5, 8, False, 'good') for demand in demand_levels]
    off_peak_prices = [model.predict_price(demand, 0.5, 14, False, 'good') for demand in demand_levels]
    
    plt.plot(demand_levels, peak_hour_prices, 'g-', label='Peak Hour (8 AM)')
    plt.plot(demand_levels, off_peak_prices, 'y-', label='Off-Peak (2 PM)')
    plt.title('Price Multipliers vs Demand Level')
    plt.xlabel('Demand Level (Normalized)')
    plt.ylabel('Price Multiplier')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 3. Price multipliers for different supply levels (peak hour, weekday)
    plt.subplot(2, 2, 3)
    supply_levels = np.linspace(0, 1, 11)
    supply_prices = [model.predict_price(0.8, supply, 8, False, 'good') for supply in supply_levels]
    
    plt.plot(supply_levels, supply_prices, 'm-')
    plt.title('Price Multipliers vs Supply Level (Peak Hour)')
    plt.xlabel('Supply Level (Normalized)')
    plt.ylabel('Price Multiplier')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Price multipliers for different weather conditions
    plt.subplot(2, 2, 4)
    weather_conditions = ['good', 'moderate', 'bad']
    weather_prices_peak = [model.predict_price(0.8, 0.5, 8, False, weather) for weather in weather_conditions]
    weather_prices_offpeak = [model.predict_price(0.5, 0.7, 14, False, weather) for weather in weather_conditions]
    
    x = range(len(weather_conditions))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], weather_prices_peak, width, label='Peak Hour (8 AM)')
    plt.bar([i + width/2 for i in x], weather_prices_offpeak, width, label='Off-Peak (2 PM)')
    plt.title('Price Multipliers for Different Weather Conditions')
    plt.xlabel('Weather Condition')
    plt.ylabel('Price Multiplier')
    plt.xticks(x, weather_conditions)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pricing_recommendations.png')
    plt.close()

# Function to create a pricing dashboard
def create_pricing_dashboard(model, ride_data, hotspots):
    """
    Create an HTML dashboard for dynamic pricing
    
    Parameters:
    -----------
    model : DynamicPricingRL
        Trained RL model
    ride_data : DataFrame
        Historical ride data
    hotspots : DataFrame
        Hotspot information
    """
    # Create directory for dashboard
    os.makedirs('pricing_dashboard', exist_ok=True)
    
    # Generate visualizations
    model.generate_price_heatmap()
    visualize_pricing_recommendations(model, ride_data, hotspots)
    
    # Create HTML file
    with open('pricing_dashboard/index.html', 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dynamic Pricing Dashboard - Namma Yatri</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin-bottom: 40px; }
                .flex-container { display: flex; flex-wrap: wrap; justify-content: space-between; }
                .chart { width: 48%; margin-bottom: 20px; }
                .full-width { width: 100%; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .price-calculator {
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }
                .form-group {
                    margin-bottom: 15px;
                }
                label {
                    display: block;
                    margin-bottom: 5px;
                    font-weight: bold;
                }
                input, select {
                    width: 100%;
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                button {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #45a049;
                }
                .result {
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #e9f7ef;
                    border-radius: 4px;
                    display: none;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Dynamic Pricing Dashboard</h1>
                    <p>AI-driven pricing recommendations for optimal supply-demand balance</p>
                </div>
                
                <div class="price-calculator">
                    <h2>Price Calculator</h2>
                    <div class="form-group">
                        <label for="location">Location:</label>
                        <select id="location">
                            <option value="0">High Demand Area</option>
                            <option value="1">Medium Demand Area</option>
                            <option value="2">Low Demand Area</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="hour">Hour of Day:</label>
                        <select id="hour">
        ''')
        
        # Add hours to dropdown
        for hour in range(24):
            f.write(f'<option value="{hour}">{hour}:00</option>\n')
        
        f.write('''
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="day-type">Day Type:</label>
                        <select id="day-type">
                            <option value="0">Weekday</option>
                            <option value="1">Weekend</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="weather">Weather Condition:</label>
                        <select id="weather">
                            <option value="good">Good</option>
                            <option value="moderate">Moderate</option>
                            <option value="bad">Bad</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="base-fare">Base Fare (INR):</label>
                        <input type="number" id="base-fare" value="50" min="10" max="500">
                    </div>
                    <button onclick="calculatePrice()">Calculate Price</button>
                    
                    <div class="result" id="price-result">
                        <h3>Recommended Price</h3>
                        <p>Base Fare: ₹<span id="result-base-fare">50</span></p>
                        <p>Price Multiplier: <span id="result-multiplier">1.5</span>x</p>
                        <p>Final Price: ₹<span id="result-final-price">75</span></p>
                        <p>Expected Driver Acceptance Rate: <span id="result-acceptance">85</span>%</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Price Multiplier Heatmap</h2>
                    <div class="full-width">
                        <img src="../price_heatmap.png" width="100%">
                    </div>
                    <p>This heatmap shows the optimal price multipliers for different combinations of demand and supply levels during morning peak hours on weekdays.</p>
                </div>
                
                <div class="section">
                    <h2>Pricing Recommendations</h2>
                    <div class="full-width">
                        <img src="../pricing_recommendations.png" width="100%">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Key Insights</h2>
                    <ul>
                        <li>During peak hours (7-10 AM, 4-8 PM), price multipliers should be higher to incentivize drivers</li>
                        <li>Weekend pricing differs from weekday pricing, with generally lower multipliers during morning hours</li>
                        <li>Bad weather conditions warrant higher price multipliers to maintain driver availability</li>
                        <li>Areas with high demand-to-supply ratios require higher multipliers to balance the market</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Recommendations for Implementation</h2>
                    <ol>
                        <li>Implement dynamic pricing gradually, starting with peak hours only</li>
                        <li>Communicate clearly to drivers how the pricing system works and how it benefits them</li>
                        <li>Provide transparency to riders about why prices may be higher during certain times</li>
                        <li>Consider offering loyalty rewards to frequent riders to offset occasional higher prices</li>
                        <li>Continuously monitor and adjust the pricing model based on driver and rider feedback</li>
                    </ol>
                </div>
                
                <div class="footer">
                    <p>© 2023 Namma Yatri Hackathon - AI-Powered Dynamic Pricing</p>
                </div>
            </div>
            
            <script>
                # Simple price calculator logic
                function calculatePrice() {
                    // This is a simplified version - in a real implementation, this would call an API
                    // that uses the actual trained model
                    
                    const location = document.getElementById('location').value;
                    const hour = parseInt(document.getElementById('hour').value);
                    const dayType = document.getElementById('day-type').value;
                    const weather = document.getElementById('weather').value;
                    const baseFare = parseFloat(document.getElementById('base-fare').value);
                    
                    // Simplified logic to mimic the model's behavior
                    let multiplier = 1.0;
                    
                    // Peak hour adjustment
                    if ((hour >= 7 && hour <= 10) || (hour >= 16 && hour <= 20)) {
                        multiplier += 0.3;
                    }
                    
                    // Weekend adjustment
                    if (dayType === '1' && hour < 12) {
                        multiplier -= 0.1;
                    }
                    
                    // Weather adjustment
                    if (weather === 'moderate') {
                        multiplier += 0.1;
                    } else if (weather === 'bad') {
                        multiplier += 0.2;
                    }
                    
                    // Location adjustment
                    if (location === '0') { // High demand
                        multiplier += 0.2;
                    } else if (location === '2') { // Low demand
                        multiplier -= 0.1;
                    }
                    
                    // Ensure multiplier is within bounds
                    multiplier = Math.max(1.0, Math.min(2.0, multiplier));
                    
                    // Calculate final price
                    const finalPrice = baseFare * multiplier;
                    
                    // Calculate expected acceptance rate
                    let acceptanceRate = 70;
                    if (multiplier >= 1.5) {
                        acceptanceRate = 85;
                    } else if (multiplier >= 1.2) {
                        acceptanceRate = 75;
                    }
                    
                    // Update the result
                    document.getElementById('result-base-fare').textContent = baseFare.toFixed(2);
                    document.getElementById('result-multiplier').textContent = multiplier.toFixed(2);
                    document.getElementById('result-final-price').textContent = finalPrice.toFixed(2);
                    document.getElementById('result-acceptance').textContent = acceptanceRate;
                    
                    // Show the result
                    document.getElementById('price-result').style.display = 'block';
                }
            </script>
        </body>
        </html>
        ''')
    
        print(f"Pricing dashboard created at 'pricing_dashboard'")

# Main execution
if __name__ == "__main__":
    import seaborn as sns
    
    print("Dynamic Pricing RL Model for Namma Yatri")
    print("=======================================")
    
    # Load ride data
    try:
        print("\nLoading ride data...")
        ride_data = pd.read_csv('ride_data.csv')
        ride_data['timestamp'] = pd.to_datetime(ride_data['timestamp'])
        
        # Load hotspot data if available, otherwise use empty DataFrame
        try:
            hotspots = pd.read_csv('hotspots.csv')
        except FileNotFoundError:
            print("Hotspots data not found. Creating empty DataFrame.")
            hotspots = pd.DataFrame(columns=['cluster', 'pickup_lat', 'pickup_lng', 'count', 'area_name'])
    except FileNotFoundError:
        print("Ride data not found. Creating synthetic data for demonstration.")
        # Create synthetic ride data for demonstration
        ride_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='H'),
            'pickup_lat': np.random.uniform(12.9, 13.1, 1000),
            'pickup_lng': np.random.uniform(77.5, 77.7, 1000),
            'driver_id': np.random.randint(1, 101, 1000),
            'fare': np.random.uniform(50, 300, 1000)
        })
        hotspots = pd.DataFrame(columns=['cluster', 'pickup_lat', 'pickup_lng', 'count', 'area_name'])
    
    # Prepare training data
    print("\nPreparing training data...")
    training_data = prepare_synthetic_data(ride_data, hotspots)
    
    # Initialize and train the model
    model = DynamicPricingRL(base_fare=50)
    model.train(training_data, episodes=500)  # Reduced episodes for faster execution
    
    # Plot training metrics
    print("\nPlotting training metrics...")
    model.plot_training_metrics()
    
    # Save the trained model
    model.save_model()
    
    # Create pricing dashboard
    print("\nCreating pricing dashboard...")
    create_pricing_dashboard(model, ride_data, hotspots)
    
    # Example predictions
    print("\nExample price predictions:")
    print("Morning peak hour (8 AM), weekday, high demand, low supply, good weather:")
    price = model.predict_price(0.9, 0.3, 8, False, 'good')
    print(f"  Price multiplier: {price:.2f}x")
    
    print("Evening peak hour (6 PM), weekday, high demand, low supply, bad weather:")
    price = model.predict_price(0.9, 0.3, 18, False, 'bad')
    print(f"  Price multiplier: {price:.2f}x")
    
    print("Off-peak hour (2 PM), weekday, medium demand, high supply, good weather:")
    price = model.predict_price(0.5, 0.8, 14, False, 'good')
    print(f"  Price multiplier: {price:.2f}x")
    
    print("Weekend morning (9 AM), medium demand, medium supply, moderate weather:")
    price = model.predict_price(0.6, 0.6, 9, True, 'moderate')
    print(f"  Price multiplier: {price:.2f}x")
    
    print("\nDynamic pricing model implementation completed!")
    print("Open 'pricing_dashboard/index.html' in a web browser to view the dashboard.")
    print("You can also use the model to make predictions for specific scenarios.")
    
    # Recommendations for driver incentives
    print("\nRecommendations for driver incentives:")
    print("1. Offer guaranteed minimum earnings for drivers who accept rides during peak hours")
    print("2. Provide bonuses for drivers who complete a certain number of rides in hotspot areas")
    print("3. Implement a streak bonus system for consecutive ride acceptances")
    print("4. Create a tiered reward system based on driver acceptance rates")
    print("5. Offer special incentives for rides during bad weather conditions")
    
    # Recommendations for passenger experience
    print("\nRecommendations for passenger experience:")
    print("1. Provide estimated wait times based on current supply and demand")
    print("2. Offer loyalty discounts to frequent riders to offset surge pricing")
    print("3. Implement a fare split feature for shared rides to reduce costs during peak hours")
    print("4. Send notifications about expected surge pricing in advance")
    print("5. Allow passengers to schedule rides during off-peak hours at lower rates")