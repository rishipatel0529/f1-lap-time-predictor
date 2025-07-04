## Examples

### Retrieve online features

```bash
python - <<EOF
from feast import FeatureStore
fs = FeatureStore(repo_path="feature_repo")
df = fs.get_online_features(
  ["tire_temp_stats:tire_temp_avg","tire_temp_stats:tire_temp_std"],
  entity_rows=[{"car_id": 2}]
).to_df()
print(df)
