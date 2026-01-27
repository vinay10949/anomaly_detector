from .NormalBehaviorSimulator import EntityProfiles, MetricGenerators, TemporalPatterns, CorrelationEngine
from .AnomalyInjector import PointAnomalies, ContextualAnomalies, CollectiveAnomalies, TimingAnomalies
from .RealismEngine import NoiseGenerator, MissingDataSimulator, DriftSimulator, SeasonalityModeler
from .OutputFormatter import JSONLWriter, CSVWriter, MetadataGenerator, StatisticsReporter
import random

class DataGenerator:
    def __init__(self, output_format='jsonl'):
        self.output_format = output_format
        self.entity_profiles = EntityProfiles()
        self.metric_gen = MetricGenerators(self.entity_profiles)
        self.temporal = TemporalPatterns()
        self.correlation = CorrelationEngine()
        self.point_anom = PointAnomalies()
        self.context_anom = ContextualAnomalies()
        self.collective_anom = CollectiveAnomalies()
        self.timing_anom = TimingAnomalies()
        self.noise_gen = NoiseGenerator()
        self.missing_sim = MissingDataSimulator()
        self.drift_sim = DriftSimulator()
        self.seasonality_mod = SeasonalityModeler()
        if output_format == 'jsonl':
            self.writer = JSONLWriter('dataset.jsonl')
        elif output_format == 'csv':
            self.writer = CSVWriter('dataset.csv')
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        self.metadata = MetadataGenerator()
        self.stats = StatisticsReporter()

    def generate_normal_data(self, entity_id, entity_type, signal_type, metric, start_time, end_time, interval=60):
        data = []
        current_time = start_time
        while current_time <= end_time:
            base_value = self.metric_gen.generate_value(entity_type, signal_type, metric)
            value = self.temporal.apply_temporal(base_value, current_time)
            value = self.noise_gen.add_noise(value)
            data.append({
                'timestamp': current_time,
                'entity_id': entity_id,
                'signal_type': signal_type,
                'metric': metric,
                'value': round(value, 2)
            })
            current_time += interval
        return data

    def inject_anomalies(self, data, entity_type=None):
        import statistics
        
        # Define probabilities for anomaly types
        anomaly_types = ['point', 'collective', 'contextual', 'timing']
        probabilities = [0.5, 0.2, 0.15, 0.15]  # 50% point, 20% collective, 15% contextual, 15% timing
        
        # Compute global_mean for contextual anomalies
        if data:
            global_mean = statistics.mean(d['value'] for d in data)
        else:
            global_mean = 0
        
        # Decide how many anomalies to inject
        num_anomalies = max(1, len(data) // 20)  # about 5%
        
        for _ in range(num_anomalies):
            # Choose anomaly type
            anomaly_type = random.choices(anomaly_types, probabilities)[0]
            
            if anomaly_type == 'point':
                # Inject point anomaly
                idx = random.randint(0, len(data)-1)
                point_methods = ['inject_spike', 'inject_drop']
                method = random.choice(point_methods)
                if method == 'inject_spike':
                    data[idx]['value'] = self.point_anom.inject_spike(data[idx]['value'])
                    anomaly_subtype = 'point_spike'
                elif method == 'inject_drop':
                    data[idx]['value'] = self.point_anom.inject_drop(data[idx]['value'], random.randint(20, 50))
                    anomaly_subtype = 'point_drop'
                self.metadata.add_anomaly(data[idx]['timestamp'], data[idx]['entity_id'], data[idx]['signal_type'], data[idx]['metric'], entity_type, anomaly_subtype)
            
            elif anomaly_type == 'collective':
                # Inject collective anomaly
                collective_methods = ['inject', 'inject_variance_explosion', 'inject_trend_reversal', 'inject_square_wave_distortion']
                method = random.choice(collective_methods)
                if method == 'inject':
                    data = self.collective_anom.inject(data)
                    anomaly_subtype = 'collective_spike_cluster'
                elif method == 'inject_variance_explosion':
                    multiplier = random.uniform(2, 5)
                    data = self.collective_anom.inject_variance_explosion(data, multiplier)
                    anomaly_subtype = 'collective_variance_explosion'
                elif method == 'inject_trend_reversal':
                    data = self.collective_anom.inject_trend_reversal(data)
                    anomaly_subtype = 'collective_trend_reversal'
                elif method == 'inject_square_wave_distortion':
                    data = self.collective_anom.inject_square_wave_distortion(data)
                    anomaly_subtype = 'collective_square_wave_distortion'
                # Add metadata for the first point
                if data:
                    self.metadata.add_anomaly(data[0]['timestamp'], data[0]['entity_id'], data[0]['signal_type'], data[0]['metric'], entity_type, anomaly_subtype)
            
            elif anomaly_type == 'contextual':
                # Inject contextual anomaly
                data = self.context_anom.inject_contextual(data, global_mean)
                anomaly_subtype = 'contextual_anomaly'
                # Add metadata for the affected point (assuming it affects one)
                # But since inject_contextual picks one, but to simplify, add for first
                if data:
                    self.metadata.add_anomaly(data[0]['timestamp'], data[0]['entity_id'], data[0]['signal_type'], data[0]['metric'], entity_type, anomaly_subtype)
            
            elif anomaly_type == 'timing':
                # Inject timing anomaly
                timing_methods = ['inject', 'inject_lag', 'inject_missed_beat']
                method = random.choice(timing_methods)
                if method == 'inject':
                    data = self.timing_anom.inject(data)
                    anomaly_subtype = 'timing_timestamp_shift'
                elif method == 'inject_lag':
                    lag_steps = random.randint(1, min(5, len(data)//2))
                    data = self.timing_anom.inject_lag(data, lag_steps)
                    anomaly_subtype = 'timing_lag'
                elif method == 'inject_missed_beat':
                    idx = random.randint(0, len(data)-1)
                    data = self.timing_anom.inject_missed_beat(data, idx)
                    anomaly_subtype = 'timing_missed_beat'
                # Add metadata
                if data:
                    self.metadata.add_anomaly(data[0]['timestamp'], data[0]['entity_id'], data[0]['signal_type'], data[0]['metric'], entity_type, anomaly_subtype)
        
        return data

    def generate_dataset(self, entities, start_time, end_time):
        all_data = []
        for entity in entities:
            for signal_type in entity['signal_types']:
                for metric in signal_type['metrics']:
                    data = self.generate_normal_data(entity['id'], entity['type'], signal_type['type'], metric, start_time, end_time)
                    data = self.inject_anomalies(data, entity['type'])
                    data = self.correlation.correlate(data)
                    data = self.missing_sim.simulate_missing(data)
                    data = self.drift_sim.simulate_drift(data)
                    data = self.seasonality_mod.model_seasonality(data)
                    all_data.extend(data)
        self.writer.write(all_data)
        self.metadata.save_metadata('metadata.json')
        self.stats.report(all_data)
        return all_data