import time

from feast import FeatureStore


def benchmark(car_id: int, runs: int = 50):
    store = FeatureStore(repo_path="feature_repo")
    # warm-up
    _ = store.get_online_features(
        features=["tire_temp_stats:tire_temp_avg"], entity_rows=[{"car_id": car_id}]
    ).to_dict()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = store.get_online_features(
            features=["tire_temp_stats:tire_temp_avg"], entity_rows=[{"car_id": car_id}]
        ).to_dict()
        times.append((time.perf_counter() - start) * 1000)

    avg_ms = sum(times) / len(times)
    print(f"Average online read latency over {runs} runs: {avg_ms:.2f} ms")


if __name__ == "__main__":
    benchmark(car_id=2)
