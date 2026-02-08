import random
import math
import time
import json
import numpy as np
from datetime import datetime, timedelta

# -----------------------------
# Configuration
# -----------------------------
START_TIMESTAMP = int(time.time()) - (3600 * 24 * 7)  # Start 7 days ago
STEP_SECONDS = 60              # 1-minute resolution
NUM_POINTS = 14 * 24 * 60       # 7 days of 1-minute data (10080 points)

ENTITY_ID = "web_server_01"
SIGNAL_TYPE = "network_event"
METRIC = "sendmsg"  # or "http_requests", "cpu_usage", etc.

# Realistic baseline values based on metric type
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

config = METRIC_CONFIGS.get(METRIC, METRIC_CONFIGS["sendmsg"])

# -----------------------------
# Realistic Patterns
# -----------------------------
def get_daily_pattern(dt):
    """Get daily pattern factor based on time of day"""
    hour = dt.hour + dt.minute/60
    
    # Business hours pattern (9am-5pm peak)
    if 9 <= hour <= 17:
        # Peak around 1-2pm, gradual ramp up/down
        if hour < 13:
            # Morning ramp up: 9am to 1pm
            progress = (hour - 9) / 4
            return 0.6 + 0.4 * progress
        else:
            # Afternoon ramp down: 1pm to 5pm
            progress = (hour - 13) / 4
            return 1.0 - 0.4 * progress
    elif 0 <= hour <= 6:
        # Night time: very low activity
        return 0.3 + 0.1 * math.sin(hour * math.pi / 3)
    else:
        # Evening: moderate activity
        return 0.5 + 0.2 * math.sin((hour - 18) * math.pi / 6)

def get_weekly_pattern(dt):
    """Get weekly pattern factor based on day of week"""
    weekday = dt.weekday()  # 0=Monday, 6=Sunday
    
    # Workdays vs weekend pattern
    if weekday < 5:  # Monday-Friday
        # Tuesday/Wednesday peak, Monday/Friday medium
        if weekday == 1 or weekday == 2:  # Tue, Wed
            return 1.0
        elif weekday == 0:  # Monday
            return 0.8
        elif weekday == 3:  # Thursday
            return 0.9
        else:  # Friday
            return 0.7
    else:  # Weekend
        if weekday == 5:  # Saturday
            return 0.4
        else:  # Sunday
            return 0.3

def get_seasonal_factor(timestamp):
    """Get seasonal/trend factor"""
    dt = datetime.fromtimestamp(timestamp)
    
    # Long-term trend (slow increase over weeks)
    days_since_start = (timestamp - START_TIMESTAMP) / (3600 * 24)
    trend = 1.0 + 0.001 * days_since_start  # 0.1% daily growth
    
    return trend

def apply_autoregressive_smoothing(values, alpha=0.85):
    """Apply autoregressive smoothing for realistic continuity"""
    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed_value = alpha * smoothed[-1] + (1 - alpha) * values[i]
        smoothed.append(smoothed_value)
    return smoothed

def inject_realistic_noise(values, noise_level):
    """Add realistic noise with some correlation"""
    n = len(values)
    # AR(1) noise for more realistic correlated noise
    noise = [random.gauss(0, noise_level)]
    for i in range(1, n):
        noise.append(0.7 * noise[-1] + 0.3 * random.gauss(0, noise_level))
    return [v + n for v, n in zip(values, noise)]

# -----------------------------
# Anomaly Injection Functions
# -----------------------------
def inject_point_spike(values, idx, magnitude_multiplier):
    """Inject a realistic point spike"""
    baseline = np.median(values[max(0, idx-10):min(len(values), idx+10)])
    spike_height = baseline * magnitude_multiplier
    values[idx] += spike_height + random.gauss(0, spike_height * 0.2)
    return values

def inject_gradual_drift(values, start_idx, end_idx):
    """Inject a gradual drift anomaly"""
    drift_length = end_idx - start_idx
    for i in range(start_idx, end_idx):
        progress = (i - start_idx) / drift_length
        # Sigmoid-shaped drift for realism
        drift_amount = config["baseline"] * 0.5 * (1 / (1 + math.exp(-10 * (progress - 0.5))))
        values[i] += drift_amount
    return values

def inject_collective_anomaly(values, start_idx, end_idx):
    """Inject a collective anomaly (cluster of high values)"""
    cluster_length = end_idx - start_idx
    # Varying intensity within the cluster
    for i in range(start_idx, end_idx):
        position = (i - start_idx) / cluster_length
        # Bell-shaped intensity
        intensity = math.exp(-8 * (position - 0.5) ** 2)
        anomaly_amount = config["baseline"] * 2.0 * intensity
        values[i] += anomaly_amount * random.uniform(0.8, 1.2)
    return values

def inject_contextual_anomaly(values, start_idx, end_idx):
    """Inject contextual anomaly (abnormal for time of day)"""
    for i in range(start_idx, end_idx):
        dt = datetime.fromtimestamp(START_TIMESTAMP + i * STEP_SECONDS)
        hour = dt.hour
        
        # If it's nighttime (1-5am), make it look like daytime traffic
        if 1 <= hour <= 5:
            values[i] *= random.uniform(3.0, 5.0)
        # If it's daytime, make it look like nighttime
        elif 10 <= hour <= 16:
            values[i] *= random.uniform(0.1, 0.3)
    return values

def inject_sudden_drop(values, start_idx, duration_minutes=30):
    """Inject sudden service drop"""
    end_idx = start_idx + duration_minutes
    for i in range(start_idx, min(end_idx, len(values))):
        # Drop to very low values
        values[i] *= random.uniform(0.05, 0.15)
        # Add some recovery toward the end
        if i > start_idx + duration_minutes * 0.7:
            recovery = (i - (start_idx + duration_minutes * 0.7)) / (duration_minutes * 0.3)
            values[i] *= 1.0 + recovery * 0.5
    return values

def inject_oscillating_anomaly(values, start_idx, duration_minutes=60):
    """Inject oscillating/bouncing anomaly"""
    end_idx = start_idx + duration_minutes
    for i in range(start_idx, min(end_idx, len(values))):
        position = (i - start_idx) / duration_minutes
        # Sine wave oscillation
        oscillation = math.sin(position * 4 * math.pi) * 0.5 + 0.5
        values[i] *= 1.0 + oscillation * random.uniform(1.5, 2.5)
    return values

# -----------------------------
# Data generation
# -----------------------------
data = []
raw_values = []

print(f"Generating {NUM_POINTS} points of {METRIC} data...")
print(f"Configuration: {config}")

# First pass: generate base pattern
for i in range(NUM_POINTS):
    timestamp = START_TIMESTAMP + i * STEP_SECONDS
    dt = datetime.fromtimestamp(timestamp)
    
    # Calculate pattern factors
    daily_factor = get_daily_pattern(dt)
    weekly_factor = get_weekly_pattern(dt)
    seasonal_factor = get_seasonal_factor(timestamp)
    
    # Base value with patterns
    value = (
        config["baseline"] 
        * daily_factor 
        * weekly_factor 
        * seasonal_factor
    )
    
    # Add some random variation
    variation = random.gauss(0, config["noise_std"] * 0.5)
    value += variation
    
    raw_values.append(value)

# Apply smoothing for realistic continuity
smoothed_values = apply_autoregressive_smoothing(raw_values, alpha=0.88)

# Add correlated noise
final_values = inject_realistic_noise(smoothed_values, config["noise_std"])

# -----------------------------
# Inject realistic anomalies
# -----------------------------
print("Injecting realistic anomalies...")

# 1. Point spikes (sudden traffic surges)
spike_indices = random.sample(range(100, NUM_POINTS - 100), 3)
for idx in spike_indices:
    final_values = inject_point_spike(final_values, idx, config["anomaly_spike_multiplier"])

# 2. Collective anomaly (DDoS-like sustained attack)
collective_start = random.randint(2000, 3000)
collective_end = collective_start + random.randint(30, 120)  # 30-120 minutes
final_values = inject_collective_anomaly(final_values, collective_start, collective_end)

# 3. Gradual drift (resource leak or memory buildup)
drift_start = random.randint(4000, 5000)
drift_end = drift_start + random.randint(180, 360)  # 3-6 hours
final_values = inject_gradual_drift(final_values, drift_start, drift_end)

# 4. Contextual anomaly (unusual nighttime activity)
context_start = random.randint(6000, 7000)
context_end = context_start + random.randint(60, 180)
final_values = inject_contextual_anomaly(final_values, context_start, context_end)

# 5. Sudden drop (service outage)
drop_start = random.randint(8000, 8500)
final_values = inject_sudden_drop(final_values, drop_start, duration_minutes=45)

# 6. Oscillating anomaly (bouncing service)
osc_start = random.randint(9000, 9500)
final_values = inject_oscillating_anomaly(final_values, osc_start, duration_minutes=90)

# -----------------------------
# Final processing and bounds checking
# -----------------------------
for i in range(NUM_POINTS):
    # Ensure values stay within realistic bounds
    bounded_value = max(config["min_value"], min(config["max_value"], final_values[i]))
    
    # Round to reasonable precision
    bounded_value = round(bounded_value, 2)
    
    timestamp = START_TIMESTAMP + i * STEP_SECONDS
    
    record = {
        "timestamp": int(timestamp),
        "entity_id": ENTITY_ID,
        "signal_type": SIGNAL_TYPE,
        "metric": METRIC,
        "value": float(bounded_value)
    }
    
    data.append(record)

# -----------------------------
# Save to file
# -----------------------------
filename = f"{METRIC}_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
with open(filename, "w") as f:
    for row in data:
        f.write(json.dumps(row) + "\n")

# Create a simple metadata file
metadata = {
    "generated_at": datetime.now().isoformat(),
    "metric": METRIC,
    "entity_id": ENTITY_ID,
    "signal_type": SIGNAL_TYPE,
    "num_points": NUM_POINTS,
    "step_seconds": STEP_SECONDS,
    "start_timestamp": START_TIMESTAMP,
    "end_timestamp": START_TIMESTAMP + (NUM_POINTS * STEP_SECONDS),
    "config_used": config,
    "anomalies_injected": {
        "point_spikes": spike_indices,
        "collective_anomaly": [collective_start, collective_end],
        "gradual_drift": [drift_start, drift_end],
        "contextual_anomaly": [context_start, context_end],
        "sudden_drop": drop_start,
        "oscillating_anomaly": osc_start
    }
}

with open(f"{filename}.metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Generated {len(data)} records")
print(f"Data saved to: {filename}")
print(f"Metadata saved to: {filename}.metadata.json")
print(f"Value range: {min(d['value'] for d in data):.2f} - {max(d['value'] for d in data):.2f}")
print(f"Mean value: {sum(d['value'] for d in data)/len(data):.2f}")

# Optional: Create a simple visualization
try:
    import matplotlib.pyplot as plt
    values = [d['value'] for d in data]
    timestamps = [d['timestamp'] for d in data]
    
    plt.figure(figsize=(15, 6))
    plt.plot(timestamps, values, 'b-', alpha=0.7, linewidth=0.8)
    plt.xlabel('Timestamp')
    plt.ylabel(f'{METRIC} Value')
    plt.title(f'Realistic Time Series: {METRIC}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=150)
    print(f"Plot saved to: {filename}.png")
except ImportError:
    print("Matplotlib not installed, skipping visualization")