import math
import random
import json
from collections import deque

class FeatureExtractionLayer:
    def __init__(self):
        self.history = deque(maxlen=10)  # Keep recent values for temporal features

    def extract_features(self, data_point):
        """
        Extract features from a single data point.
        """
        value = data_point['value']
        timestamp = data_point['timestamp']

        # Basic value encoding
        encoded_value = self.encode_value(value)

        # Temporal features
        temporal_features = self.extract_temporal_features(value)

        # Entity context (simplified)
        entity_context = self.get_entity_context(data_point)

        features = {
            'encoded_value': encoded_value,
            'temporal_features': temporal_features,
            'entity_context': entity_context
        }

        # Update history
        self.history.append(value)

        return features

    def encode_value(self, value):
        # Simple encoding: normalize to 0-1 range (assuming range 0-1000)
        return max(0, min(1, value / 1000))

    def extract_temporal_features(self, value):
        if len(self.history) < 2:
            return [0, 0, 0]  # diff, trend, variance

        recent = list(self.history)
        diff = value - recent[-2] if len(recent) >= 2 else 0
        trend = sum(recent[-3:]) / len(recent[-3:]) if len(recent) >= 3 else value
        variance = sum((x - trend) ** 2 for x in recent[-5:]) / len(recent[-5:]) if len(recent) >= 5 else 0

        return [diff, trend, variance]

    def get_entity_context(self, data_point):
        # Simplified: hash of entity_id and metric
        context = hash(f"{data_point['entity_id']}_{data_point['metric']}") % 100
        return context / 100  # Normalize

class PreprocessingLayer:
    def __init__(self):
        self.mean = 0
        self.std = 1

    def fit(self, data):
        values = [d['value'] for d in data]
        self.mean = sum(values) / len(values)
        self.std = math.sqrt(sum((x - self.mean) ** 2 for x in values) / len(values))

    def transform(self, features):
        # Normalize features
        normalized = {}
        for key, value in features.items():
            if isinstance(value, list):
                normalized[key] = [(v - self.mean) / self.std for v in value]
            else:
                normalized[key] = (value - self.mean) / self.std
        return normalized

class LSMPathway:
    def __init__(self, reservoir_size=500, input_size=10):
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.reservoir_state = [0.0] * reservoir_size
        # Random reservoir connections
        self.reservoir_weights = [[random.gauss(0, 0.1) for _ in range(reservoir_size)] for _ in range(reservoir_size)]
        # Input to reservoir connections
        self.input_weights = [[random.gauss(0, 0.1) for _ in range(reservoir_size)] for _ in range(input_size)]
        self.readout_weights = None
        self.training_targets = []

    def train(self, training_data):
        print("Training LSM: Reservoir dynamics and readout")
        # Train reservoir on normal data
        reservoir_states = []
        for features in training_data:
            input_vector = self.flatten_features(features)
            self.update_reservoir(input_vector)
            reservoir_states.append(self.reservoir_state.copy())

        # Train readout for anomaly detection (autoencoder style)
        # Target is the input, so train to reconstruct
        targets = [self.flatten_features(f) for f in training_data]
        self.train_readout(reservoir_states, targets)

    def process_input(self, features):
        input_vector = self.flatten_features(features)
        self.update_reservoir(input_vector)

        if self.readout_weights:
            # Compute reconstruction error as anomaly score
            reconstructed = self.readout(self.reservoir_state)
            target = input_vector + [0] * (len(reconstructed) - len(input_vector))  # Pad
            error = sum((r - t) ** 2 for r, t in zip(reconstructed[:len(target)], target)) / len(target)
            temporal_score = error
        else:
            temporal_score = 0

        return temporal_score

    def update_reservoir(self, input_vector):
        # Update reservoir neurons
        new_state = [0.0] * self.reservoir_size
        for i in range(self.reservoir_size):
            reservoir_input = sum(self.reservoir_state[j] * self.reservoir_weights[j][i] for j in range(self.reservoir_size))
            input_contrib = sum(input_vector[j] * self.input_weights[j][i] for j in range(min(len(input_vector), self.input_size)))
            new_state[i] = math.tanh(reservoir_input + input_contrib + self.reservoir_state[i] * 0.9)  # Leaky

        self.reservoir_state = new_state

    def flatten_features(self, features):
        flat = []
        for key, value in features.items():
            if isinstance(value, list):
                flat.extend(value)
            else:
                flat.append(value)
        return flat[:self.input_size]

    def readout(self, reservoir_state):
        if not self.readout_weights:
            return []
        return [sum(reservoir_state[j] * self.readout_weights[i][j] for j in range(self.reservoir_size)) for i in range(self.input_size)]

    def train_readout(self, reservoir_states, targets):
        # Simple linear regression for readout (without numpy)
        # For now, keep random weights (actual training requires more complex implementation)
        self.readout_weights = [[random.random() for _ in range(self.reservoir_size)] for _ in range(self.input_size)]

class HTMPathway:
    def __init__(self):
        try:
            import nupic
            from nupic.algorithms.spatial_pooler import SpatialPooler
            from nupic.algorithms.temporal_memory import TemporalMemory
            from nupic.algorithms.anomaly import Anomaly
            from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder

            self.nupic_available = True
            self.num_features = 5  # Number of features we encode
            self.sdr_size = 100  # SDR size per feature
            self.total_input_size = self.num_features * self.sdr_size
            self.num_columns = 2048

            # Initialize SDR Encoder
            self.encoder = RandomDistributedScalarEncoder(
                resolution=0.01,
                w=21,
                n=self.sdr_size,
                name="feature_encoder",
                seed=42
            )

            # Initialize NuPIC HTM components
            self.spatial_pooler = SpatialPooler(
                inputDimensions=(self.total_input_size,),
                columnDimensions=(self.num_columns,),
                potentialRadius=self.total_input_size // 2,
                potentialPct=0.5,
                globalInhibition=True,
                localAreaDensity=-1.0,
                numActiveColumnsPerInhArea=40,
                stimulusThreshold=0,
                synPermInactiveDec=0.008,
                synPermActiveInc=0.05,
                synPermConnected=0.1,
                minPctOverlapDutyCycles=0.001,
                dutyCyclePeriod=1000,
                boostStrength=0.0,
                seed=42,
                spVerbosity=0
            )

            self.temporal_memory = TemporalMemory(
                columnDimensions=(self.num_columns,),
                cellsPerColumn=32,
                activationThreshold=13,
                initialPermanence=0.21,
                connectedPermanence=0.5,
                minThreshold=10,
                maxNewSynapseCount=20,
                permanenceIncrement=0.1,
                permanenceDecrement=0.1,
                predictedSegmentDecrement=0.0,
                seed=42
            )

            self.anomaly = Anomaly(
                slidingWindowSize=100,
                mode='pure',
                binaryAnomalyThreshold=0.5
            )

            print("NuPIC HTM components with SDR encoder initialized")

        except ImportError:
            print("NuPIC not available, using simplified HTM with SDR-like encoding")
            self.nupic_available = False
            # Fallback to simplified implementation
            self.encoder = None
            self.spatial_pooler = None
            self.temporal_memory = None
            self.anomaly = None

    def train(self, training_data):
        if not self.nupic_available:
            print("NuPIC not available, skipping HTM training")
            return

        print("Training NuPIC HTM with SDR encoding on", len(training_data), "samples")
        for features in training_data:
            sdr = self.encode_features(features)

            # Ensure SDR is the correct size
            if len(sdr) != self.total_input_size:
                sdr = sdr[:self.total_input_size] + [0] * (self.total_input_size - len(sdr))

            # Spatial Pooling
            active_columns = [0] * self.num_columns
            self.spatial_pooler.compute(sdr, True, active_columns)

            # Temporal Memory
            self.temporal_memory.compute(self.num_columns, active_columns, True)

            # Anomaly learning
            anomaly_score = self.anomaly.compute(active_columns, None, None)

    def process_input(self, features):
        if not self.nupic_available:
            # Fallback simplified scoring
            return random.random()

        sdr = self.encode_features(features)

        # Ensure SDR is the correct size
        if len(sdr) != self.total_input_size:
            sdr = sdr[:self.total_input_size] + [0] * (self.total_input_size - len(sdr))

        # Spatial Pooling
        active_columns = [0] * self.num_columns
        self.spatial_pooler.compute(sdr, False, active_columns)

        # Temporal Memory
        self.temporal_memory.compute(self.num_columns, active_columns, False)

        # Anomaly detection
        anomaly_score = self.anomaly.compute(active_columns, None, None)

        return anomaly_score

    def encode_features(self, features):
        if self.nupic_available and self.encoder:
            # Use SDR encoder for each feature
            sdr_list = []
            for key, value in sorted(features.items()):
                if isinstance(value, list):
                    for v in value:
                        encoded = self.encoder.encode(v)
                        sdr_list.extend(encoded)
                else:
                    encoded = self.encoder.encode(value)
                    sdr_list.extend(encoded)
            return sdr_list
        else:
            # Fallback: simple binary encoding
            pattern = []
            for key, value in sorted(features.items()):
                if isinstance(value, list):
                    for v in value:
                        pattern.append(1 if v > 0.5 else 0)
                else:
                    pattern.append(1 if value > 0.5 else 0)
            return pattern

class FusionLayer:
    def __init__(self, htm_weight=0.6, lsm_weight=0.4):
        self.htm_weight = htm_weight
        self.lsm_weight = lsm_weight

    def fuse_scores(self, htm_score, lsm_score):
        # Weighted average
        combined_score = self.htm_weight * htm_score + self.lsm_weight * lsm_score

        # Confidence (simplified)
        confidence = min(1, combined_score * 2)

        # Voting (simplified)
        anomaly_flag = combined_score > 0.5

        return {
            'score': combined_score,
            'confidence': confidence,
            'anomaly_flag': anomaly_flag
        }

class OutputLayer:
    def __init__(self):
        pass

    def generate_output(self, fusion_result, data_point):
        score = fusion_result['score']
        confidence = fusion_result['confidence']
        anomaly_flag = fusion_result['anomaly_flag']

        explanation = self.generate_explanation(score, data_point)

        return {
            'anomaly_flag': anomaly_flag,
            'score': score,
            'confidence': confidence,
            'explanation': explanation,
            'data_point': data_point
        }

    def generate_explanation(self, score, data_point):
        if score > 0.7:
            return f"High anomaly score ({score:.2f}) for {data_point['entity_id']} {data_point['metric']} at {data_point['timestamp']}"
        elif score > 0.5:
            return f"Moderate anomaly score ({score:.2f}) detected"
        else:
            return "Normal behavior"

class AnomalyDetector:
    def __init__(self):
        self.feature_extractor = FeatureExtractionLayer()
        self.preprocessor = PreprocessingLayer()
        self.lsm_pathway = LSMPathway()
        self.htm_pathway = HTMPathway()
        self.fusion_layer = FusionLayer()
        self.output_layer = OutputLayer()
        self.trained = False

    def train(self, training_data):
        print("Starting anomaly detector training...")
        # Fit preprocessor on training data
        self.preprocessor.fit(training_data)

        # Prepare training features
        training_features = []
        for data_point in training_data[:500]:  # Use subset for training
            features = self.feature_extractor.extract_features(data_point)
            normalized_features = self.preprocessor.transform(features)
            training_features.append(normalized_features)

        # Train HTM
        self.htm_pathway.train(training_features)

        # Train LSM
        self.lsm_pathway.train(training_features)

        self.trained = True
        print("Anomaly detector fully trained on", len(training_features), "samples")

    def detect_anomaly(self, data_point):
        if not self.trained:
            raise ValueError("Detector must be trained before detection")

        # Feature extraction
        features = self.feature_extractor.extract_features(data_point)

        # Preprocessing
        normalized_features = self.preprocessor.transform(features)

        # LSM pathway
        lsm_score = self.lsm_pathway.process_input(normalized_features)

        # HTM pathway
        htm_score = self.htm_pathway.process_input(normalized_features)

        # Fusion
        fusion_result = self.fusion_layer.fuse_scores(htm_score, lsm_score)

        # Output
        output = self.output_layer.generate_output(fusion_result, data_point)

        return output

# Example usage
if __name__ == "__main__":
    # Load some sample data
    from DataGenerator import DataGenerator

    dg = DataGenerator()
    # Use existing dataset
    try:
        with open('dataset.jsonl', 'r') as f:
            sample_data = []
            for i, line in enumerate(f):
                if i >= 100: break  # Use first 100 points
                sample_data.append(json.loads(line))

        detector = AnomalyDetector()
        detector.train(sample_data)

        # Test on a few points
        for point in sample_data[:5]:
            result = detector.detect_anomaly(point)
            print(f"Point: {point['entity_id']} {point['metric']} - Anomaly: {result['anomaly_flag']}, Score: {result['score']:.3f}")

    except FileNotFoundError:
        print("Dataset not found. Generate data first.")