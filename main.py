import argparse
import time
from DataGenerator import DataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic anomaly dataset")
    parser.add_argument(
        "--points",
        type=int,
        default=None,
        help="Exact number of data points to generate across all streams",
    )
    parser.add_argument(
        "--anomaly-percent",
        type=float,
        default=5.0,
        help="Anomaly percentage to inject (must be between 5 and 15)",
    )
    parser.add_argument(
        "--duration-hours",
        type=int,
        default=24,
        help="Time-range duration in hours (used only when --points is not set)",
    )
    parser.add_argument(
        "--output-format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Dataset output format",
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        default=0.9,
        help="Continuity smoothing factor between 0 and 0.99 (higher is smoother)",
    )
    parser.add_argument(
        "--series-style",
        choices=["default", "taxi_like"],
        default="taxi_like",
        help="Shape style for baseline time series",
    )
    args = parser.parse_args()
    if args.anomaly_percent < 5.0 or args.anomaly_percent > 15.0:
        parser.error("--anomaly-percent must be between 5 and 15")
    if args.points is not None and args.points <= 0:
        parser.error("--points must be > 0")
    if args.duration_hours <= 0:
        parser.error("--duration-hours must be > 0")
    if args.smoothness < 0 or args.smoothness >= 1:
        parser.error("--smoothness must be in [0, 1)")
    return args

# Example usage
if __name__ == "__main__":
    args = parse_args()
    dg = DataGenerator(output_format=args.output_format)
    dg.continuity_alpha = args.smoothness
    dg.series_style = args.series_style
    start_time = int(time.time())
    end_time = start_time + (3600 * args.duration_hours)
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
    if args.points is not None:
        data = dg.generate_dataset(
            entities,
            start_time,
            target_points=args.points,
            anomaly_percent=args.anomaly_percent,
        )
    else:
        data = dg.generate_dataset(
            entities,
            start_time,
            end_time=end_time,
            anomaly_percent=args.anomaly_percent,
        )

    print(
        f"Generated {len(data)} data points "
        f"(anomaly_percent={args.anomaly_percent:.2f}%, style={args.series_style}, output={args.output_format})"
    )
