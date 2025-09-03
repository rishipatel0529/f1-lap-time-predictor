"""
scripts/driver_telemetry_code_files/kafka_telemetry_consumer.py

Consumes live telemetry messages from Kafka and writes features to a Feast online store.
Each Kafka record is expected to contain tire temperature statistics per driver.
The script maps the JSON payload into a small Pandas DataFrame and ingests it into the
"tire_temp_stats" FeatureView, timestamped with the current UTC event time.
"""
import json
from datetime import datetime

import pandas as pd
from feast import FeatureStore
from kafka import KafkaConsumer

# Initialize Feast by pointing at the local feature repository (defines entities/views)
store = FeatureStore(repo_path="feature_repo")

# Subscribe to the telemetry topic; messages are JSON-decoded into Python dicts
consumer = KafkaConsumer(
    "f1-telemetry",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)

try:
    # Stream loop: read each Kafka message, transform to DataFrame, and write to Feast
    for msg in consumer:
        data = msg.value

        # Build a one-row frame with entity key, event time, and feature values
        df = pd.DataFrame(
            [
                {
                    "driver_id": data.get("driver_id", 0),
                    "event_time": datetime.utcnow(),
                    "tire_temp_avg": data.get("tire_temp_avg", 0.0),
                    "tire_temp_std": data.get("tire_temp_std", 0.0),
                }
            ]
        )

        # Defensive: replace any missing values to keep ingestion schema-stable
        df = df.fillna(0.0)

        # Write features into the online store for low-latency serving
        store.write_to_online_store(feature_view_name="tire_temp_stats", df=df)
        print(f"Written to Feast: {df}")

except KeyboardInterrupt:
    # shutdown when the user interrupts the process
    print("Stopping stream ingestion.")
finally:
    # Ensure the Kafka consumer is closed cleanly
    consumer.close()
