import subprocess
from datetime import datetime

from feast import FeatureStore


def run_f1_demo():
    store = FeatureStore(repo_path=".")

    print("\n--- Run feast apply ---")
    subprocess.run(["feast", "apply"])

    print("\n--- Try fetching online features ---")
    fetch_online_features(store)

    print("\n--- Materialize incremental data ---")
    store.materialize_incremental(end_date=datetime.now())

    print("\n--- Online features after materialization ---")
    fetch_online_features(store)


def fetch_online_features(store):
    entity_rows = [
        {
            "car_id": 1,
        },
        {
            "car_id": 2,
        },
    ]

    features_to_fetch = [
        "tire_temp_stats:tire_temp_avg",
        "tire_temp_stats:tire_temp_std",
    ]

    result = store.get_online_features(
        features=features_to_fetch, entity_rows=entity_rows
    ).to_dict()

    for key, value in sorted(result.items()):
        print(key, ":", value)


if __name__ == "__main__":
    run_f1_demo()
