from DataGenerator import DataGenerator
import time

# Example usage
if __name__ == "__main__":
    # Choose output format: 'jsonl' or 'csv'
    output_format = 'jsonl'  # Change to 'csv' for CSV output
    dg = DataGenerator(output_format=output_format)
    start_time = int(time.time())
    end_time = start_time + 3600 * 24  # 1 day for better patterns
    entities = [
        {
            'id': 'web_server_01',
            'type': 'web_server',
            'signal_types': [
                {
                    'type': 'network_event',
                    'metrics': ['http_requests', 'sendmsg', 'recvmsg']
                },
                {
                    'type': 'system',
                    'metrics': ['cpu_usage', 'memory_usage']
                }
            ]
        },
        {
            'id': 'db_server_01',
            'type': 'database_server',
            'signal_types': [
                {
                    'type': 'database',
                    'metrics': ['disk_io', 'query_count', 'connection_pool', 'lock_waits']
                }
            ]
        },
        {
            'id': 'app_server_01',
            'type': 'application_server',
            'signal_types': [
                {
                    'type': 'application',
                    'metrics': ['gc_time', 'thread_count', 'heap_usage', 'api_latency']
                }
            ]
        },
        {
            'id': 'edge_device_01',
            'type': 'edge_device',
            'signal_types': [
                {
                    'type': 'sensor',
                    'metrics': ['temperature', 'packet_loss', 'battery_level', 'sensor_reading']
                }
            ]
        },
        {
            'id': 'network_infra_01',
            'type': 'network_infrastructure',
            'signal_types': [
                {
                    'type': 'network',
                    'metrics': ['bandwidth_in', 'bandwidth_out', 'packet_drops', 'latency']
                }
            ]
        }
    ]
    data = dg.generate_dataset(entities, start_time, end_time)
    print(f"Generated {len(data)} data points")