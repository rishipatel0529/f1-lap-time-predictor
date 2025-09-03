"""
scripts/driver_telemetry_code_files/kafka_telemetry_producer.py

Produces telemetry messages to a Kafka topic from a historical CSV.
Reads rows from data/driver_telemetry_csv_files/historical_telemetry.csv,
serializes each record as JSON, and publishes to the "f1-telemetry" topic
with a small delay to simulate a live stream.
"""

import json
import time

import pandas as pd
from kafka import KafkaProducer

# Configure Kafka producer and JSON-encode each payload before sending
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

# Load historical telemetry to replay; each row becomes one Kafka message
df = pd.read_csv("data/driver_telemetry_csv_files/historical_telemetry.csv")

# Iterate row-by-row, convert to dict, and publish to the telemetry topic
for _, row in df.iterrows():
    message = row.to_dict()
    print(f"Sending message: {message}")
    producer.send("f1-telemetry", message)
    time.sleep(0.1) # throttle to avoid overwhelming consumers; simulates 10 Hz

# Ensure all buffered messages are delivered to the broker before exiting
producer.flush()

# Simple completion signal for the operator/logs
print("All messages sent!")
