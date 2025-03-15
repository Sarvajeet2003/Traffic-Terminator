import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import uuid

class WaitingQueueSystem:
    def __init__(self, max_queue_time=30):
        """
        Initialize the waiting queue system
        
        Parameters:
        -----------
        max_queue_time : int
            Maximum time in minutes a user can be in the queue
        """
        self.queue = {}  # Dictionary to store queue entries by location
        self.user_requests = {}  # Dictionary to track user requests
        self.max_queue_time = max_queue_time
        
        # Load queue data if available
        self.load_queue()
    
    def add_to_queue(self, user_id, pickup_lat, pickup_lng, destination_lat, destination_lng, 
                    scheduled_time=None, area_name=None):
        """
        Add a user to the waiting queue
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user
        pickup_lat : float
            Pickup latitude
        pickup_lng : float
            Pickup longitude
        destination_lat : float
            Destination latitude
        destination_lng : float
            Destination longitude
        scheduled_time : datetime, optional
            Scheduled pickup time (if None, assumed to be immediate)
        area_name : str, optional
            Name of the pickup area if known
            
        Returns:
        --------
        dict
            Queue entry information including estimated wait time
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Current time
        current_time = datetime.now()
        
        # If scheduled time is not provided, set to current time (immediate request)
        if scheduled_time is None:
            scheduled_time = current_time
        
        # Calculate location key (grid cell)
        location_key = self._get_location_key(pickup_lat, pickup_lng)
        
        # Initialize queue for this location if it doesn't exist
        if location_key not in self.queue:
            self.queue[location_key] = []
        
        # Estimate wait time based on queue length and location demand
        estimated_wait_time = self._estimate_wait_time(location_key, scheduled_time)
        
        # Create queue entry
        queue_entry = {
            'request_id': request_id,
            'user_id': user_id,
            'pickup_lat': pickup_lat,
            'pickup_lng': pickup_lng,
            'destination_lat': destination_lat,
            'destination_lng': destination_lng,
            'area_name': area_name,
            'request_time': current_time,
            'scheduled_time': scheduled_time,
            'estimated_wait_time': estimated_wait_time,
            'status': 'waiting',
            'priority': self._calculate_priority(current_time, scheduled_time)
        }
        
        # Add to queue
        self.queue[location_key].append(queue_entry)
        
        # Sort queue by priority
        self.queue[location_key] = sorted(self.queue[location_key], 
                                         key=lambda x: x['priority'], 
                                         reverse=True)
        
        # Add to user requests
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        self.user_requests[user_id].append(request_id)
        
        # Save queue
        self.save_queue()
        
        return queue_entry
    
    def get_queue_status(self, request_id=None, user_id=None):
        """
        Get status of a specific request or all requests for a user
        
        Parameters:
        -----------
        request_id : str, optional
            Specific request ID to check
        user_id : str, optional
            User ID to check all requests for
            
        Returns:
        --------
        dict or list
            Queue entry or list of entries
        """
        if request_id:
            # Find the specific request
            for location_key, queue_entries in self.queue.items():
                for entry in queue_entries:
                    if entry['request_id'] == request_id:
                        return entry
            return None
        
        elif user_id:
            # Find all requests for this user
            if user_id not in self.user_requests:
                return []
            
            user_entries = []
            for request_id in self.user_requests[user_id]:
                entry = self.get_queue_status(request_id=request_id)
                if entry:
                    user_entries.append(entry)
            
            return user_entries
        
        else:
            # Return all queue entries
            all_entries = []
            for location_key, queue_entries in self.queue.items():
                all_entries.extend(queue_entries)
            
            return all_entries
    
    def update_queue_entry(self, request_id, status=None, driver_id=None):
        """
        Update a queue entry
        
        Parameters:
        -----------
        request_id : str
            Request ID to update
        status : str, optional
            New status ('waiting', 'matched', 'completed', 'cancelled')
        driver_id : str, optional
            Driver ID if matched
            
        Returns:
        --------
        dict
            Updated queue entry
        """
        for location_key, queue_entries in self.queue.items():
            for i, entry in enumerate(queue_entries):
                if entry['request_id'] == request_id:
                    if status:
                        self.queue[location_key][i]['status'] = status
                    
                    if driver_id:
                        self.queue[location_key][i]['driver_id'] = driver_id
                        self.queue[location_key][i]['matched_time'] = datetime.now()
                    
                    # Save queue
                    self.save_queue()
                    
                    return self.queue[location_key][i]
        
        return None
    
    def cancel_request(self, request_id):
        """
        Cancel a queue request
        
        Parameters:
        -----------
        request_id : str
            Request ID to cancel
            
        Returns:
        --------
        bool
            True if cancelled successfully, False otherwise
        """
        for location_key, queue_entries in self.queue.items():
            for i, entry in enumerate(queue_entries):
                if entry['request_id'] == request_id:
                    # Update status to cancelled
                    self.queue[location_key][i]['status'] = 'cancelled'
                    
                    # Save queue
                    self.save_queue()
                    
                    return True
        
        return False
    
    def get_next_in_queue(self, driver_lat, driver_lng, max_distance=5.0):
        """
        Get the next request in queue for a driver
        
        Parameters:
        -----------
        driver_lat : float
            Driver's current latitude
        driver_lng : float
            Driver's current longitude
        max_distance : float
            Maximum distance in kilometers to consider
            
        Returns:
        --------
        dict
            Next queue entry for the driver
        """
        # Calculate driver's location key
        driver_location_key = self._get_location_key(driver_lat, driver_lng)
        
        # Get nearby location keys (including current location)
        nearby_keys = self._get_nearby_grid_cells(driver_lat, driver_lng)
        
        # Find the highest priority waiting request in nearby locations
        best_entry = None
        best_priority = -float('inf')
        
        for key in nearby_keys:
            if key in self.queue:
                for entry in self.queue[key]:
                    if entry['status'] == 'waiting':
                        # Calculate distance between driver and pickup
                        distance = self._calculate_distance(
                            driver_lat, driver_lng, 
                            entry['pickup_lat'], entry['pickup_lng']
                        )
                        
                        if distance <= max_distance and entry['priority'] > best_priority:
                            best_entry = entry
                            best_priority = entry['priority']
        
        return best_entry
    
    def _get_nearby_grid_cells(self, lat, lng, radius=1):
        """
        Get nearby grid cells based on latitude and longitude
        
        Parameters:
        -----------
        lat : float
            Latitude
        lng : float
            Longitude
        radius : int
            Number of grid cells to consider in each direction
            
        Returns:
        --------
        list
            List of nearby grid cell keys
        """
        precision = 3  # Same precision as used in _get_location_key
        
        # Calculate the approximate degree change for the given radius
        # At the equator, 1 degree is approximately 111 km
        # This is a simplification and will be less accurate at higher latitudes
        degree_change = radius * (0.001 / 111)  # Convert to degrees
        
        nearby_keys = []
        
        # Generate grid cells in a square around the current location
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                nearby_lat = round(lat + i * degree_change, precision)
                nearby_lng = round(lng + j * degree_change, precision)
                nearby_keys.append(f"{nearby_lat},{nearby_lng}")
        
        return nearby_keys
    
    def clean_expired_entries(self):
        """
        Clean expired entries from the queue
        
        Returns:
        --------
        int
            Number of entries cleaned
        """
        current_time = datetime.now()
        cleaned_count = 0
        
        for location_key in list(self.queue.keys()):
            updated_queue = []
            
            for entry in self.queue[location_key]:
                # Skip entries that are already matched or completed
                if entry['status'] in ['matched', 'completed']:
                    updated_queue.append(entry)
                    continue
                
                # Check if entry has expired
                request_time = entry['request_time']
                if (current_time - request_time).total_seconds() / 60 > self.max_queue_time:
                    # Mark as expired
                    entry['status'] = 'expired'
                    cleaned_count += 1
                
                updated_queue.append(entry)
            
            self.queue[location_key] = updated_queue
        
        # Save queue if any entries were cleaned
        if cleaned_count > 0:
            self.save_queue()
        
        return cleaned_count
    
    def save_queue(self):
        """Save queue data to file"""
        queue_data = {
            'queue': self.queue,
            'user_requests': self.user_requests,
            'timestamp': datetime.now()
        }
        
        try:
            with open('waiting_queue_data.pkl', 'wb') as f:
                pickle.dump(queue_data, f)
        except Exception as e:
            print(f"Error saving queue data: {e}")
    
    def load_queue(self):
        """Load queue data from file"""
        try:
            if os.path.exists('waiting_queue_data.pkl'):
                with open('waiting_queue_data.pkl', 'rb') as f:
                    queue_data = pickle.load(f)
                
                self.queue = queue_data.get('queue', {})
                self.user_requests = queue_data.get('user_requests', {})
                
                # Clean expired entries
                self.clean_expired_entries()
        except Exception as e:
            print(f"Error loading queue data: {e}")
            self.queue = {}
            self.user_requests = {}
    
    def _get_location_key(self, lat, lng, precision=3):
        """
        Convert latitude and longitude to a grid cell key
        
        Parameters:
        -----------
        lat : float
            Latitude
        lng : float
            Longitude
        precision : int
            Decimal precision for the grid
            
        Returns:
        --------
        str
            Location key
        """
        # Round coordinates to create a grid cell
        lat_rounded = round(lat, precision)
        lng_rounded = round(lng, precision)
        
        return f"{lat_rounded},{lng_rounded}"
    
    def _estimate_wait_time(self, location_key, scheduled_time):
        """
        Estimate wait time for a new request
        
        Parameters:
        -----------
        location_key : str
            Location grid cell key
        scheduled_time : datetime
            Scheduled pickup time
            
        Returns:
        --------
        int
            Estimated wait time in minutes
        """
        # Count waiting requests in this location
        waiting_count = sum(1 for entry in self.queue.get(location_key, []) 
                          if entry['status'] == 'waiting')
        
        # Base wait time based on queue length
        base_wait = waiting_count * 3  # Assume 3 minutes per request in queue
        
        # Adjust for time of day (peak hours)
        hour = scheduled_time.hour
        if (7 <= hour <= 10) or (16 <= hour <= 20):
            # Peak hours - longer wait times
            base_wait = base_wait * 1.5
        
        # Add some randomness
        wait_time = int(base_wait + np.random.normal(0, 2))
        
        # Ensure wait time is positive
        return max(1, wait_time)
    
    def _calculate_priority(self, request_time, scheduled_time):
        """
        Calculate priority score for queue sorting
        
        Parameters:
        -----------
        request_time : datetime
            Time when request was made
        scheduled_time : datetime
            Scheduled pickup time
            
        Returns:
        --------
        float
            Priority score (higher = higher priority)
        """
        current_time = datetime.now()
        
        # Time until scheduled pickup
        time_until_scheduled = (scheduled_time - current_time).total_seconds() / 60
        
        # How long the request has been in the queue
        time_in_queue = (current_time - request_time).total_seconds() / 60
        
        # Priority increases as scheduled time approaches and the longer it's been in queue
        if time_until_scheduled <= 0:
            # Immediate or past scheduled time gets highest priority
            time_factor = 100
        else:
            # Priority increases as scheduled time approaches
            time_factor = 100 / (1 + time_until_scheduled)
        
        # Queue time factor - the longer in queue, the higher priority
        queue_factor = min(50, time_in_queue / 2)
        
        return time_factor + queue_factor
    
    def _calculate_distance(self, lat1, lng1, lat2, lng2):
        """
        Calculate distance between two points in kilometers
        
        Parameters:
        -----------
        lat1, lng1 : float
            Coordinates of first point
        lat2, lng2 : float
            Coordinates of second point
            
        Returns:
        --------
        float
            Distance in kilometers
        """
        # Simple Euclidean distance for demonstration
        # In a real implementation, use Haversine formula or a mapping API
        return ((lat1 - lat2) ** 2 + (lng1 - lng2) ** 2) ** 0.5 * 111  # Rough conversion to km

# Function to get estimated wait time for a location
def get_estimated_wait_time(lat, lng, scheduled_time=None):
    """
    Get estimated wait time for a location without adding to queue
    
    Parameters:
    -----------
    lat : float
        Pickup latitude
    lng : float
        Pickup longitude
    scheduled_time : datetime, optional
        Scheduled pickup time
        
    Returns:
    --------
    int
        Estimated wait time in minutes
    """
    queue_system = WaitingQueueSystem()
    location_key = queue_system._get_location_key(lat, lng)
    
    if scheduled_time is None:
        scheduled_time = datetime.now()
    
    return queue_system._estimate_wait_time(location_key, scheduled_time)


def get_queue_statistics(self):
        """
        Get statistics about the current queue state
        
        Returns:
        --------
        dict
            Dictionary containing queue statistics
        """
        all_entries = self.get_queue_status()
        
        # Initialize statistics
        stats = {
            'total_requests': len(all_entries),
            'waiting_requests': 0,
            'matched_requests': 0,
            'completed_requests': 0,
            'cancelled_requests': 0,
            'expired_requests': 0,
            'avg_wait_time': 0,
            'avg_priority': 0,
            'locations': {},
            'hourly_distribution': {hour: 0 for hour in range(24)}
        }
        
        if not all_entries:
            return stats
        
        # Calculate statistics
        wait_times = []
        priorities = []
        
        for entry in all_entries:
            # Count by status
            status = entry['status']
            if status in ['waiting', 'matched', 'completed', 'cancelled', 'expired']:
                stats[f'{status}_requests'] += 1
            
            # Collect wait times and priorities
            wait_times.append(entry['estimated_wait_time'])
            priorities.append(entry['priority'])
            
            # Count by location
            location_key = self._get_location_key(entry['pickup_lat'], entry['pickup_lng'])
            if location_key not in stats['locations']:
                stats['locations'][location_key] = 0
            stats['locations'][location_key] += 1
            
            # Count by hour
            hour = entry['request_time'].hour
            stats['hourly_distribution'][hour] += 1
        
        # Calculate averages
        if wait_times:
            stats['avg_wait_time'] = sum(wait_times) / len(wait_times)
        if priorities:
            stats['avg_priority'] = sum(priorities) / len(priorities)
        
        # Find busiest locations (top 5)
        stats['busiest_locations'] = sorted(
            stats['locations'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Find peak hours (top 3)
        stats['peak_hours'] = sorted(
            stats['hourly_distribution'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        return stats


def optimize_queue(self, available_drivers=0):
        """
        Optimize the queue based on current demand and driver availability
        
        Parameters:
        -----------
        available_drivers : int
            Number of available drivers in the system
            
        Returns:
        --------
        dict
            Optimization results and recommendations
        """
        # Get current queue statistics
        stats = self.get_queue_statistics()
        
        # Initialize optimization results
        optimization = {
            'recommendations': [],
            'driver_allocation': {},
            'estimated_clearance_time': 0
        }
        
        # If no waiting requests, no optimization needed
        if stats['waiting_requests'] == 0:
            optimization['recommendations'].append("No waiting requests in queue. No optimization needed.")
            return optimization
        
        # Calculate driver deficit/surplus
        driver_deficit = stats['waiting_requests'] - available_drivers
        
        if driver_deficit > 0:
            optimization['recommendations'].append(
                f"Driver shortage: Need {driver_deficit} more drivers to clear the queue."
            )
        else:
            optimization['recommendations'].append(
                f"Driver surplus: Have {abs(driver_deficit)} extra drivers available."
            )
        
        # Estimate time to clear the queue
        if available_drivers > 0:
            # Assuming each driver can handle one request every 15 minutes on average
            clearance_time = (stats['waiting_requests'] / available_drivers) * 15
            optimization['estimated_clearance_time'] = clearance_time
            
            optimization['recommendations'].append(
                f"Estimated time to clear queue: {clearance_time:.1f} minutes."
            )
        
        # Allocate drivers to busiest locations
        if available_drivers > 0 and stats['busiest_locations']:
            total_requests = sum(count for _, count in stats['busiest_locations'])
            
            for location, count in stats['busiest_locations']:
                # Allocate drivers proportionally to request count
                allocated_drivers = int((count / total_requests) * available_drivers)
                optimization['driver_allocation'][location] = allocated_drivers
                
                optimization['recommendations'].append(
                    f"Allocate {allocated_drivers} drivers to location {location} ({count} requests)."
                )
        
        # Recommendations based on peak hours
        if stats['peak_hours']:
            peak_hour_msg = "Current peak hours: "
            for hour, count in stats['peak_hours']:
                peak_hour_msg += f"{hour}:00-{hour+1}:00 ({count} requests), "
            
            optimization['recommendations'].append(peak_hour_msg[:-2])
        
        return optimization