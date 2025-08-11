import os
import sys
from contextlib import contextmanager
import logging
import absl.logging
import warnings

# Suppress TensorFlow and C++ backend logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Suppress absl logging
absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.set_stderrthreshold(absl.logging.FATAL)

# Suppress strided slice warnings
warnings.filterwarnings('ignore', message='Index out of range using input dim')

# Context manager for stderr suppression
@contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
tf.get_logger().setLevel(logging.FATAL)
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.debugging.disable_check_numerics()

# Suppress third-party logs
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)

import gin
from absl import app
from absl import flags
from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory as trajectory_lib
from tf_agents.trajectories import time_step as ts
from src.runtimes.ecomm_runtime import ECommRuntime
from src.stories import ecomm_story
from src.core.registry import REGISTRY
from src.core.registry import register
import src.agents.random_agent
from src.agents.slateq_agent import SlateQAgent 
from src.metrics.ranking_metrics import ndcg_at_k, slate_mrr
from src.metrics.logger import MetricsLogger

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('gin_files', [], 'Paths to the config files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin parameter bindings.')

@register("slateq")
def make_slateq(time_step_spec, action_spec, **kwargs):
    return SlateQAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        num_users=kwargs.get("num_users"),
        num_topics=kwargs.get("num_topics", 10),
        slate_size=kwargs.get("slate_size"),
        num_items=kwargs.get("num_items"),
    )

def main(argv):
    # parse gin
    gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)

    # pick agent: positional arg after flags; default "slateq"
    agent_name = argv[1] if len(argv) > 1 else "slateq"
    if agent_name not in REGISTRY:
        raise ValueError(f"Unknown agent '{agent_name}'. Options: {list(REGISTRY)}")

    # env/story config (keep your current numbers or pull from gin if you wish)
    num_users = 10
    slate_size = 5
    num_items = 100
    num_episodes = 300
    steps_per_episode = 200

    network = ecomm_story(num_users=num_users, num_items=num_items, slate_size=slate_size)
    rt = ECommRuntime(network=network)

    # action spec (same as before)
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(slate_size,),
        dtype=tf.int32,
        minimum=0,
        maximum=num_items - 1
    )

    # build a time_step_spec consistent with your loop
    agent_time_step_spec = ts.TimeStep(
        step_type=tf.TensorSpec(shape=(), dtype=tf.int32),
        reward=tf.TensorSpec(shape=(), dtype=tf.float32),
        discount=tf.TensorSpec(shape=(), dtype=tf.float32),
        observation={"interest": tf.TensorSpec(shape=(10,), dtype=tf.float32)},
    )

    # instantiate agent through registry
    agent = REGISTRY[agent_name](
        time_step_spec=agent_time_step_spec,
        action_spec=action_spec,
        num_users=num_users,
        slate_size=slate_size,
        num_items=num_items,
        num_topics=10,  # keep your current value
    )

    # warm up policy call (works for both agents)
    dummy_ts = ts.TimeStep(
        step_type=tf.zeros((num_users,), tf.int32),
        reward=tf.zeros((num_users,), tf.float32),
        discount=tf.ones((num_users,), tf.float32),
        observation={"interest": tf.zeros((num_users, 10), tf.float32)},
    )
    _ = agent.policy.action(dummy_ts)

    # If the agent learns, set up replay + dataset
    if getattr(agent, "is_learning", True):
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=num_users,
            max_length=10000,
        )
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=32,
            num_steps=2,
            single_deterministic_pass=False
        ).prefetch(3)
        dataset_iter = None  # will be made per episode

    logger = MetricsLogger(base_dir="logs")

    for episode in range(num_episodes):
        episode_losses = []
        loss_info = None

        time_step = rt.reset()
        episode_reward = 0.0
        last_slate = None
        last_choice = None
        last_reward = None

        if getattr(agent, "is_learning", True):
            dataset_iter = iter(dataset)

        for step in range(steps_per_episode):
            clean_time_step = time_step._replace(
                observation={"interest": time_step.observation["interest"]}
            )
            action_step = agent.collect_policy.action(clean_time_step)
            action = action_step.action
            next_time_step = rt.step(action)

            last_slate = action.numpy()
            last_reward = next_time_step.reward
            last_choice = next_time_step.observation["choice"]
            episode_reward += float(tf.reduce_sum(next_time_step.reward).numpy())

            # write transition only if learning
            if getattr(agent, "is_learning", True):
                cleaned_next = next_time_step._replace(
                    observation={"interest": next_time_step.observation["interest"]}
                )
                cleaned_curr = time_step._replace(
                    observation={"interest": time_step.observation["interest"]}
                )
                exp = trajectory_lib.from_transition(cleaned_curr, action_step, cleaned_next)
                replay_buffer.add_batch(exp)

                # train after warmup
                if replay_buffer.num_frames().numpy() >= 64:
                    try:
                        experience, _ = next(dataset_iter)
                    except StopIteration:
                        dataset_iter = iter(dataset)
                        experience, _ = next(dataset_iter)
                    loss_info = agent.train(experience)
                    episode_losses.append(float(loss_info.loss.numpy()))
                    if step % 1000 == 0:
                        print(f"[Episode {episode}] Step {step} | Loss: {loss_info.loss.numpy():.4f}")

                # per-step epsilon decay if available
                if hasattr(agent.collect_policy, "decay_epsilon"):
                    agent.collect_policy.decay_epsilon(steps=1)

            time_step = next_time_step

        # Ranking metrics for the *last* slate
        relevance = np.zeros_like(last_slate)
        for i in range(num_users):
            idx = last_choice[i]
            if 0 <= idx < slate_size:
                relevance[i, idx] = last_reward[i]

        ndcg = ndcg_at_k(last_slate, relevance, k=slate_size)
        mrr = slate_mrr(last_slate, relevance, k=slate_size)
        avg_loss = float(np.mean(episode_losses)) if episode_losses else None

        logger.log({
            "episode": episode,
            "total_reward": float(episode_reward),
            "loss": avg_loss,
            "ndcg@5": float(ndcg),
            "slate_mrr": float(mrr)
        })
        print(f"[Episode {episode}] Total Reward: {episode_reward:.2f}")

    trajectory_result = rt.trajectory()
    user_state = trajectory_result.get('user_state')
    if user_state:
        interest = user_state.get('interest')
        if isinstance(interest, tf.Tensor):
            for i, value in enumerate(interest.numpy()):
                print(f"User {i} final interest: {value}")
        else:
            print("Interest:", interest)
    else:
        print("No user_state in final trajectory")

    # Auto-plot at the end
    logger.close()
    from pathlib import Path
    metrics_df = pd.read_csv(logger.csv_file)
    run_name = logger.log_dir.name
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Force correct dtype
    metrics_df["episode"] = metrics_df["episode"].astype(int)
    metrics_df["total_reward"] = pd.to_numeric(metrics_df["total_reward"], errors='coerce')
    metrics_df["loss"] = pd.to_numeric(metrics_df.get("loss"), errors='coerce')
    metrics_df["ndcg@5"] = pd.to_numeric(metrics_df.get("ndcg@5"), errors='coerce')
    metrics_df["slate_mrr"] = pd.to_numeric(metrics_df.get("slate_mrr"), errors='coerce')

    # Use plain background
    plt.style.use("default")

    # Determine tick positions every 30 episodes
    xticks = metrics_df["episode"][metrics_df["episode"] % 30 == 0]

    # Plot 1: Total Reward
    plt.figure()
    plt.plot(metrics_df["episode"], metrics_df["total_reward"], label="Total Reward", color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward over Episodes")
    plt.xticks(xticks)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.savefig(plots_dir / f"{run_name}_reward.png")
    plt.close()

    # Plot 2: Loss
    if "loss" in metrics_df and metrics_df["loss"].notnull().any():
        plt.figure()
        plt.plot(metrics_df["episode"], metrics_df["loss"], label="Loss", color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Training Loss over Episodes")
        plt.xticks(xticks)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.savefig(plots_dir / f"{run_name}_loss.png")
        plt.close()

    # Plot 3: Ranking Metrics
    has_ndcg = "ndcg@5" in metrics_df and metrics_df["ndcg@5"].notnull().any()
    has_mrr = "slate_mrr" in metrics_df and metrics_df["slate_mrr"].notnull().any()

    if has_ndcg or has_mrr:
        plt.figure()
        if has_ndcg:
            plt.plot(metrics_df["episode"], metrics_df["ndcg@5"], label="NDCG@5", color='blue')
        if has_mrr:
            plt.plot(metrics_df["episode"], metrics_df["slate_mrr"], label="Slate MRR", color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Ranking Score")
        plt.title("Ranking Metrics over Episodes")
        plt.xticks(xticks)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.savefig(plots_dir / f"{run_name}_ranking.png")
        plt.close()

if __name__ == '__main__':
    app.run(main)
