import mlflow
import ray
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.logger import JsonLoggerCallback

# ─── Register your custom env ────────────────────────────────────────────────
from src.models.f1_env import env_creator


def wrapped_env_creator(config):
    # create your base env
    env = env_creator(config)
    # normalize observations & rewards
    env = NormalizeObservation(env)
    env = NormalizeReward(env)
    return env


tune.register_env("f1-pit-env", wrapped_env_creator)
# ─────────────────────────────────────────────────────────────────────────────

# 1) Initialize Ray
ray.init(ignore_reinit_error=True)

# 2) MLflow experiment setup
mlflow.set_experiment("f1_rl_week9")
ENABLE_MLFLOW = False


def train_fn(config, checkpoint_dir=None):
    # optionally track MLflow
    if ENABLE_MLFLOW:
        mlflow.start_run()
        mlflow.log_params(config)

    trainer = PPO(
        env="f1-pit-env",
        config={
            **config,
            # PPO-specific tweaks
            "clip_param": 0.2,
            "lambda": 0.95,
            "vf_loss_coeff": 1.0,
            "entropy_coeff": 0.01,
        },
    )

    for i in range(config["train_iterations"]):
        result = trainer.train()

        # first iteration: show available keys
        if i == 0:
            print("env_runners keys:", list(result["env_runners"].keys()))

        # pick the best mean-return metric
        if "episode_return_mean/default_agent" in result["env_runners"]:
            key = "episode_return_mean/default_agent"
        else:
            key = "episode_return_mean"  # fallback
        mean_return = result["env_runners"][key]

        if ENABLE_MLFLOW:
            mlflow.log_metrics({"mean_return": mean_return}, step=i)

        print(f"Iter {i:03d} mean_return={mean_return:.2f}")

    # save final checkpoint
    chk = trainer.save()
    if ENABLE_MLFLOW:
        mlflow.log_artifact(chk)
        mlflow.end_run()

    return result


if __name__ == "__main__":
    tune.run(
        train_fn,
        name="f1_rl_tuning",
        # use TensorBoard via JSON logs
        callbacks=[JsonLoggerCallback()],
        config={
            "env": "f1-pit-env",
            # Longer run for convergence
            "train_iterations": 300,
            # parallel workers for more data per iteration
            "num_workers": 1,
            "framework": "torch",
            # hyperparameter grid
            "lr": tune.grid_search([1e-3, 5e-4, 1e-4]),
            "sgd_minibatch_size": tune.grid_search([64, 256, 512]),
            # curriculum: start small and grow
            "max_laps": tune.choice([10, 30, 70]),
        },
        stop={"training_iteration": 300},
    )
