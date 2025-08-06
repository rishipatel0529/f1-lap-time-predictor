import mlflow
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.logger import JsonLoggerCallback, TBXLoggerCallback

# ─── Register your custom env ────────────────────────────────────────────────
from src.models.f1_env import env_creator

tune.register_env("f1-pit-env", env_creator)
# ─────────────────────────────────────────────────────────────────────────────

# 1) Initialize Ray
ray.init(local_mode=True, ignore_reinit_error=True)

# 2) MLflow experiment setup (only used if ENABLE_MLFLOW=True)
mlflow.set_experiment("f1_rl_week9")
ENABLE_MLFLOW = False


def train_fn(config, checkpoint_dir=None):
    trainer = PPO(env="f1-pit-env", config=config)

    # ─── main training loop ───────────────────────────────
    for i in range(config["train_iterations"]):
        result = trainer.train()

        # on the very first iteration, show what keys you actually got
        if i == 0:
            print("env_runners keys:", list(result["env_runners"].keys()))

        # pick the most likely metric key
        if "episode_return_mean/default_agent" in result["env_runners"]:
            metric_key = "episode_return_mean/default_agent"
        elif "episode_return_mean/default_policy" in result["env_runners"]:
            metric_key = "episode_return_mean/default_policy"
        else:
            # fallback: find any float-valued field
            metric_key, _ = next(
                (
                    (k, v)
                    for k, v in result["env_runners"].items()
                    if isinstance(v, float)
                ),
                (None, None),
            )

        mean_return = result["env_runners"][metric_key]

        # optional MLflow logging
        if ENABLE_MLFLOW:
            mlflow.log_metrics({"mean_return": mean_return}, step=i)

        print(f"Iter {i:03d} mean_return={mean_return:.2f}")
    # ───────────────────────────────────────────────────────

    return result


if __name__ == "__main__":
    tune.run(
        train_fn,
        name="f1_rl_tuning",
        callbacks=[
            JsonLoggerCallback(),  # still keeps your JSON backups
            TBXLoggerCallback(),  # <-- writes real TensorBoard events
        ],
        config={
            "env": "f1-pit-env",
            "train_iterations": 50,
            "num_workers": 1,
            "framework": "torch",
            "sgd_minibatch_size": tune.grid_search([64, 128]),
            "lr": tune.grid_search([1e-3, 5e-4]),
        },
        stop={"training_iteration": 50},
    )
