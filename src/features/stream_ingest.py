import json
from datetime import datetime

import pandas as pd
from feast import FeatureStore
from kafka import KafkaConsumer

store = FeatureStore(repo_path="feature_repo")

consumer = KafkaConsumer(
    "f1-telemetry",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)

try:
    for msg in consumer:
        data = msg.value

        df = pd.DataFrame(
            [
                {
                    "car_id": data.get("car_id", 0),
                    "event_time": datetime.utcnow(),
                    "tire_temp_avg": data.get("tire_temp_avg", 0.0),
                    "tire_temp_std": data.get("tire_temp_std", 0.0),
                }
            ]
        )

        df = df.fillna(0.0)

        store.write_to_online_store(feature_view_name="tire_temp_stats", df=df)
        print(f"✅ Written to Feast: {df}")

except KeyboardInterrupt:
    print("Stopping stream ingestion.")
finally:
    consumer.close()
