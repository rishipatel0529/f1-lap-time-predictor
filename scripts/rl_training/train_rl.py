import mlflow
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

# ─── Register your custom env ────────────────────────────────────────────────
from src.models.f1_env import env_creator

tune.register_env("f1-pit-env", env_creator)
# ─────────────────────────────────────────────────────────────────────────────

# 1) Initialize Ray
ray.init(local_mode=True, ignore_reinit_error=True)

# 2) MLflow experiment setup
mlflow.set_experiment("f1_rl_week9")

ENABLE_MLFLOW = False


def train_fn(config, checkpoint_dir=None):
    if ENABLE_MLFLOW:
        with mlflow.start_run():
            mlflow.log_params(config)

    trainer = PPO(env="f1-pit-env", config=config)

    for i in range(config["train_iterations"]):
        result = trainer.train()
        if ENABLE_MLFLOW:
            mlflow.log_metrics(
                {
                    "episode_reward_mean": result["episode_reward_mean"],
                    "episode_len_mean": result["episode_len_mean"],
                },
                step=i,
            )
        print(f"Iter {i:03d} reward={result['episode_reward_mean']:.2f}")

    chk = trainer.save()
    if ENABLE_MLFLOW:
        mlflow.log_artifact(chk)
    return result


tune.run(
    train_fn,
    name="f1_rl_tuning",
    config={
        "env": "f1-pit-env",
        "train_iterations": 5,  # debug small number first
        "num_workers": 0,  # one process
        "framework": "torch",
        "sgd_minibatch_size": tune.grid_search([64, 128]),
        "lr": tune.grid_search([1e-3, 5e-4]),
    },
    stop={"training_iteration": 5},
)
