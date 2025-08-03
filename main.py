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
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory as trajectory_lib
from tf_agents.trajectories import time_step as ts
from src.runtimes.ecomm_runtime import ECommRuntime
from src.stories import ecomm_story
from src.agents.slateq_agent import SlateQAgent
from src.metrics.ranking_metrics import ndcg_at_k, slate_mrr
from src.metrics.logger import MetricsLogger

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('gin_files', [], 'Paths to the config files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin parameter bindings.')

def main(argv):
    del argv
    gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)

    num_users = 10
    slate_size = 5
    num_episodes = 10

    network = ecomm_story(num_users=num_users, num_items=100, slate_size=slate_size)
    rt = ECommRuntime(network=network)

    slate_spec = network.invariants()['rec_state'].get('slate')
    action_spec = slate_spec

    agent_time_step_spec = ts.TimeStep(
        step_type=tf.TensorSpec(shape=(num_users,), dtype=tf.int32),
        reward=tf.TensorSpec(shape=(num_users,), dtype=tf.float32),
        discount=tf.TensorSpec(shape=(num_users,), dtype=tf.float32),
        observation={
            "interest": tf.TensorSpec(shape=(num_users, 10), dtype=tf.float32)
        }
    )

    agent = SlateQAgent(
        time_step_spec=agent_time_step_spec,
        action_spec=action_spec,
        num_users=num_users,
        num_topics=10,
        slate_size=slate_size,
        num_items=100
    )

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=1,
        max_length=10000
    )

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=32,
        num_steps=2,
        single_deterministic_pass=False
    ).prefetch(3)

    logger = MetricsLogger(base_dir="logs")
    loss_info = None
    last_loss = None

    for episode in range(num_episodes):
        time_step = rt.reset()
        episode_reward = 0.0
        last_slate = None
        last_choice = None
        last_reward = None

        for step in range(100):
            clean_time_step = time_step._replace(
                observation={"interest": time_step.observation["interest"]}
            )

            action_step = agent.collect_policy.action(clean_time_step)
            action = action_step.action
            next_time_step = rt.step(action)

            last_slate = action.numpy()
            last_reward = next_time_step.reward
            last_choice = next_time_step.observation["choice"]

            episode_reward += tf.reduce_sum(next_time_step.reward).numpy()

            cleaned_next_time_step = next_time_step._replace(
                observation={"interest": next_time_step.observation["interest"]}
            )

            cleaned_time_step = time_step._replace(
                observation={"interest": time_step.observation["interest"]}
            )

            experience = trajectory_lib.from_transition(cleaned_time_step, action_step, cleaned_next_time_step)
            experience = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), experience)
            replay_buffer.add_batch(experience)

            time_step = next_time_step

            if replay_buffer.num_frames().numpy() > 32:
                for experience, _ in dataset.take(1):
                    loss_info = agent.train(experience)
                    last_loss = float(loss_info.loss.numpy())

            if step % 1000 == 0 and loss_info:
                print(f"[Episode {episode}] Step {step} | Loss: {loss_info.loss.numpy():.4f}")


        # Relevance for ranking metrics
        relevance = np.zeros_like(last_slate)
        for i in range(num_users):
            idx = last_choice[i]
            if 0 <= idx < slate_size:
                relevance[i, idx] = last_reward[i]

        ndcg = ndcg_at_k(last_slate, relevance, k=5)
        mrr = slate_mrr(last_slate, relevance, k=5)

        logger.log({
            "episode": episode,
            "total_reward": float(episode_reward),
            "loss": last_loss,
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

    # Plot 1: Reward & Loss
    plt.figure()
    plt.plot(metrics_df["episode"], metrics_df["total_reward"], label="Total Reward")
    if "loss" in metrics_df:
        plt.plot(metrics_df["episode"], metrics_df["loss"], label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Reward and Loss over Episodes")
    plt.legend()
    plt.savefig(plots_dir / f"{run_name}_reward_loss.png")
    plt.close()

    # Plot 2: NDCG@5 & Slate MRR
    plt.figure()
    if "ndcg@5" in metrics_df:
        plt.plot(metrics_df["episode"], metrics_df["ndcg@5"], label="NDCG@5")
    if "slate_mrr" in metrics_df:
        plt.plot(metrics_df["episode"], metrics_df["slate_mrr"], label="Slate MRR")
    plt.xlabel("Episode")
    plt.ylabel("Ranking Score")
    plt.title("Ranking Metrics over Episodes")
    plt.legend()
    plt.savefig(plots_dir / f"{run_name}_ranking.png")
    plt.close()

if __name__ == '__main__':
    app.run(main)
