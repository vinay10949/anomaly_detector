class EntityProfiles:
    def __init__(self):
        self.profiles = {
            'web_server': {
                'network_event': {
                    'http_requests': {'mean': 100, 'std': 10},  # Low variance, predictable
                    'sendmsg': {'mean': 50, 'std': 5},
                    'recvmsg': {'mean': 60, 'std': 8}
                },
                'system': {
                    'cpu_usage': {'mean': 40, 'std': 5},
                    'memory_usage': {'mean': 60, 'std': 10}
                }
            },
            'database_server': {
                'database': {
                    'disk_io': {'mean': 80, 'std': 25},  # Bursty
                    'query_count': {'mean': 200, 'std': 50},
                    'connection_pool': {'mean': 50, 'std': 15},
                    'lock_waits': {'mean': 5, 'std': 3}
                }
            },
            'application_server': {
                'application': {
                    'gc_time': {'mean': 10, 'std': 5},  # Medium variance
                    'thread_count': {'mean': 100, 'std': 20},
                    'heap_usage': {'mean': 70, 'std': 15},
                    'api_latency': {'mean': 50, 'std': 10}
                }
            },
            'edge_device': {
                'sensor': {
                    'temperature': {'mean': 25, 'std': 2},  # Stable with drift
                    'packet_loss': {'mean': 1, 'std': 0.5},
                    'battery_level': {'mean': 80, 'std': 5},
                    'sensor_reading': {'mean': 100, 'std': 10}
                }
            },
            'network_infrastructure': {
                'network': {
                    'bandwidth_in': {'mean': 500, 'std': 100},  # High variance
                    'bandwidth_out': {'mean': 450, 'std': 90},
                    'packet_drops': {'mean': 5, 'std': 3},
                    'latency': {'mean': 20, 'std': 10}
                }
            }
        }

    def get_profile(self, entity_type, signal_type, metric):
        return self.profiles.get(entity_type, {}).get(signal_type, {}).get(metric, {})