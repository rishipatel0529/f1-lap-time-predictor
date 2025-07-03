import json
import time

import pandas as pd
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

df = pd.read_csv("data/historical_telemetry.csv")

for _, row in df.iterrows():
    message = row.to_dict()
    print(f"Sending message: {message}")
    producer.send("f1-telemetry", message)
    time.sleep(0.1)

producer.flush()

print("All messages sent!")
