import mlflow
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

from src.models.f1_env import env_creator

# Register custom env so every Ray worker can find it
tune.register_env("f1-pit-env", env_creator)

# Initialize Ray in local debug mode
ray.init(local_mode=True, ignore_reinit_error=True)

# MLflow experiment (only used if ENABLE_MLFLOW=True)
mlflow.set_experiment("f1_rl_week9")
ENABLE_MLFLOW = False


def train_fn(config, checkpoint_dir=None):
    if ENABLE_MLFLOW:
        with mlflow.start_run():
            mlflow.log_params(config)

    trainer = PPO(env="f1-pit-env", config=config)

    for i in range(config["train_iterations"]):
        result = trainer.train()

        # Debug: print out available keys on first iteration
        if i == 0:
            print("Result keys:", result.keys())
            if "metrics" in result:
                print("Nested metrics keys:", result["metrics"].keys())

        # Safely extract a reward metric
        if "episode_reward_mean" in result:
            reward = result["episode_reward_mean"]
        elif "metrics" in result and "episode_reward_mean" in result["metrics"]:
            reward = result["metrics"]["episode_reward_mean"]
        else:
            # fallback: pick the first numeric scalar
            reward = next(
                (v for v in result.values() if isinstance(v, (int, float))), None
            )

        if ENABLE_MLFLOW:
            mlflow.log_metrics({"reward": reward}, step=i)

        print(f"Iter {i:03d} reward={reward}")

    chk = trainer.save()
    if ENABLE_MLFLOW:
        mlflow.log_artifact(chk)

    return result


if __name__ == "__main__":
    tune.run(
        train_fn,
        name="f1_rl_tuning",
        config={
            "env": "f1-pit-env",
            "train_iterations": 5,  # debug small number
            "num_workers": 0,  # single process for now
            "framework": "torch",
            "sgd_minibatch_size": tune.grid_search([64, 128]),
            "lr": tune.grid_search([1e-3, 5e-4]),
        },
        stop={"training_iteration": 5},
    )
