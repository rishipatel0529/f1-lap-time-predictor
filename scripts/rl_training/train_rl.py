import mlflow
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

from src.models.f1_env import env_creator

# Register custom env
tune.register_env("f1-pit-env", env_creator)

# Initialize Ray (single-process for Colab)
ray.init(local_mode=True, ignore_reinit_error=True)

# MLflow setup (optional)
mlflow.set_experiment("f1_rl_week9")
ENABLE_MLFLOW = False


def train_fn(config, checkpoint_dir=None):
    if ENABLE_MLFLOW:
        with mlflow.start_run():
            mlflow.log_params(config)

    trainer = PPO(env="f1-pit-env", config=config)

    for i in range(config["train_iterations"]):
        result = trainer.train()
        # simple metric from the old API
        mean_ret = result["episode_reward_mean"]
        print(f"Iter {i:03d} reward={mean_ret:.2f}")

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
            "train_iterations": 5,
            "num_workers": 0,  # single process
            "framework": "torch",
            "sgd_minibatch_size": tune.grid_search([64, 128]),
            "lr": tune.grid_search([1e-3, 5e-4]),
        },
        stop={"training_iteration": 5},
    )
