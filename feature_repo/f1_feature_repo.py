from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32

car_id = Entity(name="car_id", join_keys=["car_id"], value_type=ValueType.INT64)

telemetry_source = FileSource(
    path="data/telemetry.parquet",
    timestamp_field="event_time",
)

tire_temp_stats_view = FeatureView(
    name="tire_temp_stats",
    entities=[car_id],
    ttl=timedelta(days=1),
    schema=[
        Field(name="tire_temp_avg", dtype=Float32),
        Field(name="tire_temp_std", dtype=Float32),
    ],
    online=True,
    source=telemetry_source,
)
