import random
import math
import time
import numpy as np
from datetime import datetime

class RealTimeGenerator:
    """
    Generates synthetic time-series data with realistic patterns (daily, weekly, seasonal)
    and anomalies (spikes, drifts, drops, etc.) on-the-fly.
    """
    
    # Default configuration matching gen.py
    METRIC_CONFIGS = {
        "sendmsg": {
            "baseline": 850,
            "daily_amplitude": 220,
            "weekly_factor": 1.3,
            "noise_std": 35,
            "min_value": 100,
            "max_value": 2500,
            "anomaly_spike_multiplier": 5.0,
            "anomaly_drop_multiplier": 0.1
        },
        "http_requests": {
            "baseline": 1200,
            "daily_amplitude": 400,
            "weekly_factor": 1.4,
            "noise_std": 80,
            "min_value": 150,
            "max_value": 3500,
            "anomaly_spike_multiplier": 6.0,
            "anomaly_drop_multiplier": 0.05
        },
        "cpu_usage": {
            "baseline": 45,
            "daily_amplitude": 15,
            "weekly_factor": 1.1,
            "noise_std": 5,
            "min_value": 10,
            "max_value": 95,
            "anomaly_spike_multiplier": 2.5,
            "anomaly_drop_multiplier": 0.3
        }
    }

    def __init__(self, metric="sendmsg", entity_id="web_server_01", signal_type="network_event", config_overrides=None):
        self.metric = metric
        self.entity_id = entity_id
        self.signal_type = signal_type
        
        # Load config based on metric or use defaults
        base_config = self.METRIC_CONFIGS.get(metric, self.METRIC_CONFIGS["sendmsg"])
        self.config = base_config.copy()
        if config_overrides:
            self.config.update(config_overrides)
            
        self.prev_val = self.config["baseline"]
        self.prev_noise = 0.0
        
        # Anomaly state
        self.anomaly_active = False
        self.anomaly_type = None
        self.anomaly_start_idx = 0
        self.anomaly_end_idx = 0
        self.current_idx = 0
        
        # Anomaly generator state
        self.generated_anomalies = []

    def get_daily_pattern(self, dt):
        """Get daily pattern factor based on time of day"""
        hour = dt.hour + dt.minute/60
        
        # Business hours pattern (9am-5pm peak)
        if 9 <= hour <= 17:
            # Peak around 1-2pm
            if hour < 13:
                progress = (hour - 9) / 4
                return 0.6 + 0.4 * progress
            else:
                progress = (hour - 13) / 4
                return 1.0 - 0.4 * progress
        elif 0 <= hour <= 6:
            return 0.3 + 0.1 * math.sin(hour * math.pi / 3)
        else:
            return 0.5 + 0.2 * math.sin((hour - 18) * math.pi / 6)

    def get_weekly_pattern(self, dt):
        """Get weekly pattern factor based on day of week"""
        weekday = dt.weekday()
        if weekday < 5:
            if weekday == 1 or weekday == 2: return 1.0
            elif weekday == 0: return 0.8
            elif weekday == 3: return 0.9
            else: return 0.7
        else:
            return 0.4 if weekday == 5 else 0.3

    def generate_next(self, timestamp: int) -> dict:
        """Generate the next data point for the given timestamp"""
        dt = datetime.fromtimestamp(timestamp)
        
        # 1. Base value with patterns
        daily = self.get_daily_pattern(dt)
        weekly = self.get_weekly_pattern(dt)
        
        # Autoregressive smoothing (alpha=0.88)
        base = self.config["baseline"] * daily * weekly
        smoothed_val = 0.88 * self.prev_val + (1 - 0.88) * base
        
        # 2. Correlated noise
        noise = 0.7 * self.prev_noise + 0.3 * random.gauss(0, self.config["noise_std"])
        value = smoothed_val + noise
        
        # Update state
        self.prev_val = smoothed_val
        self.prev_noise = noise
        self.current_idx += 1
        
        # 3. Inject Anomalies (Randomly trigger if none active)
        if not self.anomaly_active:
            # Random chance to start an anomaly (e.g., 5% chance per point for demo)
            if random.random() < 0.05: 
                self.start_anomaly(self.current_idx)
        
        if self.anomaly_active:
            value = self.apply_anomaly(value, self.current_idx)
            
            # Check if anomaly ended
            if self.current_idx >= self.anomaly_end_idx:
                self.anomaly_active = False
                self.anomaly_type = None

        # 4. Bounds check
        value = max(self.config["min_value"], min(self.config["max_value"], value))
        
        return {
            "timestamp": timestamp,
            "entity_id": self.entity_id,
            "signal_type": self.signal_type,
            "metric": self.metric,
            "value": float(value),
            "label": 1 if self.anomaly_active else 0
        }

    def start_anomaly(self, current_idx):
        """Randomly select and start an anomaly type"""
        types = ["spike", "collective", "drift", "drop", "contextual"]
        # Weights favors spikes and collective
        weights = [0.4, 0.2, 0.15, 0.15, 0.1] 
        self.anomaly_type = random.choices(types, weights)[0]
        self.anomaly_start_idx = current_idx
        self.anomaly_active = True
        
        if self.anomaly_type == "spike":
            self.anomaly_end_idx = current_idx + 1 # Single point
        elif self.anomaly_type == "collective":
            duration = random.randint(30, 120)
            self.anomaly_end_idx = current_idx + duration
        elif self.anomaly_type == "drift":
             duration = random.randint(100, 300)
             self.anomaly_end_idx = current_idx + duration
        elif self.anomaly_type == "drop":
             duration = random.randint(20, 50)
             self.anomaly_end_idx = current_idx + duration
        elif self.anomaly_type == "contextual":
             duration = random.randint(60, 180)
             self.anomaly_end_idx = current_idx + duration
             
        # Record for debug/visualization
        self.generated_anomalies.append({
            "type": self.anomaly_type,
            "start": self.anomaly_start_idx,
            "end": self.anomaly_end_idx
        })

    def apply_anomaly(self, value, current_idx):
        """Apply the active anomaly transformation to the value"""
        # Relative position in anomaly (0.0 to 1.0)
        duration = self.anomaly_end_idx - self.anomaly_start_idx
        if duration == 0: duration = 1
        pos = (current_idx - self.anomaly_start_idx) / duration
        
        if self.anomaly_type == "spike":
            spike_height = self.config["baseline"] * self.config["anomaly_spike_multiplier"]
            value += spike_height
            
        elif self.anomaly_type == "collective":
            # Bell-shaped intensity
            intensity = math.exp(-8 * (pos - 0.5) ** 2)
            anomaly_amount = self.config["baseline"] * 2.0 * intensity
            value += anomaly_amount * random.uniform(0.8, 1.2)
            
        elif self.anomaly_type == "drift":
            # Sigmoid drift
            drift_amount = self.config["baseline"] * 0.5 * (1 / (1 + math.exp(-10 * (pos - 0.5))))
            value += drift_amount

        elif self.anomaly_type == "drop":
            value *= random.uniform(0.05, 0.15)
            # Recovery
            if pos > 0.7:
                 recovery = (pos - 0.7) / 0.3
                 value *= 1.0 + recovery * 0.5
        
        elif self.anomaly_type == "contextual":
             value *= random.uniform(3.0, 5.0) # Assume nighttime surge for simplicity
             
        return value
