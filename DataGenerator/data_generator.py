from .NormalBehaviorSimulator import EntityProfiles, MetricGenerators, TemporalPatterns, CorrelationEngine
from .AnomalyInjector import PointAnomalies, ContextualAnomalies, CollectiveAnomalies, TimingAnomalies
from .RealismEngine import NoiseGenerator, MissingDataSimulator, DriftSimulator, SeasonalityModeler
from .OutputFormatter import JSONLWriter, MetadataGenerator, StatisticsReporter
import random

class DataGenerator:
    def __init__(self):
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
        self.writer = JSONLWriter('dataset.jsonl')
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
        # inject point anomalies randomly
        num_anomalies = max(1, len(data) // 20)  # about 5%
        for _ in range(num_anomalies):
            idx = random.randint(0, len(data)-1)
            original_value = data[idx]['value']
            anomalous_value = self.point_anom.inject_spike(original_value)
            data[idx]['value'] = anomalous_value
            self.metadata.add_anomaly(data[idx]['timestamp'], data[idx]['entity_id'], data[idx]['signal_type'], data[idx]['metric'], entity_type)
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