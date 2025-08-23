import os
import sys
from contextlib import contextmanager
import logging
import absl.logging
import warnings
import signal
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLBACKEND', 'Agg')

absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.set_stderrthreshold(absl.logging.FATAL)
warnings.filterwarnings('ignore', message='Index out of range using input dim')

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
np.random.seed(1337)
tf.random.set_seed(1337)
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

# Quiet TF logs
tf.get_logger().setLevel(logging.FATAL)
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.debugging.disable_check_numerics()
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)

with suppress_stderr():
    import matplotlib
    matplotlib.use(os.environ['MPLBACKEND'])
    import matplotlib.pyplot as plt

import pandas as pd
import gin
from absl import app
from absl import flags
from pathlib import Path

from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory as trajectory_lib
from tf_agents.trajectories import time_step as ts

from src.runtimes.ecomm_runtime import ECommRuntime
from src.stories import ecomm_story
from src.core.registry import REGISTRY, register


import src.agents.random_agent
import src.agents.greedy_agent
import src.agents.ctxbandit_agent
from src.agents.slateq_agent import SlateQAgent

from src.metrics.ranking_metrics import ndcg_at_k, slate_mrr
from src.metrics.logger import MetricsLogger

# Flags
FLAGS = flags.FLAGS
flags.DEFINE_multi_string('gin_files', [], 'Paths to config files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin parameter bindings.')

flags.DEFINE_integer('episodes', 600, 'Number of episodes.')
flags.DEFINE_integer('steps', 200, 'Steps per episode.')
flags.DEFINE_integer('num_users', 10, 'Users per batch.')
flags.DEFINE_integer('num_items', 100, 'Catalog size.')
flags.DEFINE_integer('slate_size', 5, 'Slate size K.')

flags.DEFINE_integer('warmup_frames', 5000, 'Frames before training starts.')
flags.DEFINE_integer('replay_capacity', 10000, 'Replay buffer capacity.')
flags.DEFINE_integer('batch_size', 32, 'Replay sample batch size.')

flags.DEFINE_bool('plot', True, 'Generate plots at end.')
flags.DEFINE_bool('peek_final_state', False, 'Print final user interest (can be heavy).')

# Registry wrapper for slateq
@register("slateq")
def make_slateq(time_step_spec, action_spec, **kwargs):
    return SlateQAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        num_users=kwargs.get("num_users"),
        num_topics=kwargs.get("num_topics", 10),
        slate_size=kwargs.get("slate_size"),
        num_items=kwargs.get("num_items"),
        learning_rate=kwargs.get("learning_rate", 1e-3),
        epsilon=kwargs.get("epsilon", 0.2),
        min_epsilon=kwargs.get("min_epsilon", 0.05),
        epsilon_decay_steps=kwargs.get("epsilon_decay_steps", 20000),
        target_update_period=kwargs.get("target_update_period", 1000),
        gamma=kwargs.get("gamma", 0.95),
        beta=kwargs.get("beta", 5.0),
    )

STOP = False
def _handle_exit(signum, frame):
    global STOP
    STOP = True
for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, _handle_exit)
    except Exception:
        pass

def _make_agent_timestep_spec(num_items, num_topics):
    return ts.TimeStep(
        step_type=tf.TensorSpec(shape=(), dtype=tf.int32),
        reward=tf.TensorSpec(shape=(), dtype=tf.float32),
        discount=tf.TensorSpec(shape=(), dtype=tf.float32),
        observation={
            "interest": tf.TensorSpec(shape=(num_topics,), dtype=tf.float32),
            "choice": tf.TensorSpec(shape=(), dtype=tf.int32),  # sentinel K = no-click
            "item_features": tf.TensorSpec(shape=(num_items, num_topics), dtype=tf.float32),
        },
    )

def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)

    # Agent name via argv or default
    agent_name = argv[1] if len(argv) > 1 else "slateq"
    if agent_name not in REGISTRY:
        raise ValueError(f"Unknown agent '{agent_name}'. Options: {list(REGISTRY)}")

    # Experiment knobs
    num_users = FLAGS.num_users
    slate_size = FLAGS.slate_size
    num_items = FLAGS.num_items
    num_episodes = FLAGS.episodes
    steps_per_episode = FLAGS.steps
    warmup_frames = FLAGS.warmup_frames
    replay_capacity = FLAGS.replay_capacity
    batch_size = FLAGS.batch_size
    num_topics = 10  # latent dimension used across env + agents

    # Build environment
    network = ecomm_story(num_users=num_users, num_items=num_items, slate_size=slate_size)
    rt = ECommRuntime(network=network)

    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(slate_size,), dtype=tf.int32, minimum=0, maximum=num_items - 1
    )
    agent_time_step_spec = _make_agent_timestep_spec(num_items, num_topics)

    # Instantiate agent
    agent = REGISTRY[agent_name](
        time_step_spec=agent_time_step_spec,
        action_spec=action_spec,
        num_users=num_users,
        slate_size=slate_size,
        num_items=num_items,
        num_topics=num_topics,
        epsilon_decay_steps=int(0.6 * num_episodes * steps_per_episode),
    )

    if hasattr(agent.collect_policy, "epsilon"):
        try:
            print(f"Init ε = {float(agent.collect_policy.epsilon):.3f}")
        except Exception:
            print("Init ε = (n/a)")

    # Build policy once to ensure shapes are bound
    dummy_ts = ts.TimeStep(
        step_type=tf.zeros((num_users,), tf.int32),
        reward=tf.zeros((num_users,), tf.float32),
        discount=tf.ones((num_users,), tf.float32),
        observation={
            "interest": tf.zeros((num_users, num_topics), tf.float32),
            "choice": tf.zeros((num_users,), tf.int32),
            "item_features": tf.zeros((num_users, num_items, num_topics), tf.float32),
        },
    )
    _ = agent.policy.action(dummy_ts)

    # Replay + dataset if learning
    dataset = None
    if getattr(agent, "is_learning", True):
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=num_users,
            max_length=replay_capacity,
        )
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=1,
            sample_batch_size=batch_size,
            num_steps=2,
            single_deterministic_pass=False
        ).prefetch(1)
        dataset_iter = None

    # Logger
    logger = MetricsLogger(base_dir="logs/" + agent_name)

    try:
        for episode in range(num_episodes):
            if STOP:
                print("Stop requested. Saving logs & exiting gracefully...")
                break

            episode_losses = []
            time_step = rt.reset()
            episode_reward = 0.0
            last_slate = None
            last_choice = None
            last_reward = None

            if dataset is not None:
                dataset_iter = iter(dataset)

            # Rollout
            for step in range(steps_per_episode):
                if STOP:
                    break

                action_step = agent.collect_policy.action(time_step)
                # clip indices for paranoia; also store clipped version to replay
                action = tf.clip_by_value(action_step.action, 0, num_items - 1)

                next_time_step = rt.step(action)

                # Cache last step info for metrics
                last_slate = action.numpy()                                     # [U, K]
                last_reward = next_time_step.reward.numpy()                     # [U]
                last_choice = next_time_step.observation["choice"].numpy()      # [U]
                episode_reward += float(np.sum(last_reward))

                # Learning updates
                if getattr(agent, "is_learning", True):
                    exp = trajectory_lib.from_transition(
                        time_step, action_step._replace(action=action), next_time_step
                    )
                    replay_buffer.add_batch(exp)

                    if replay_buffer.num_frames().numpy() >= warmup_frames:
                        try:
                            experience, _ = next(dataset_iter)
                        except StopIteration:
                            dataset_iter = iter(dataset)
                            experience, _ = next(dataset_iter)
                        loss_info = agent.train(experience)
                        episode_losses.append(float(loss_info.loss.numpy()))
                        if step % 500 == 0:
                            print(f"[Episode {episode}] Step {step} | Loss: {loss_info.loss.numpy():.4f}")

                    if hasattr(agent.collect_policy, "decay_epsilon"):
                        agent.collect_policy.decay_epsilon(steps=1)

                alive = rt.alive_mask()
                if not bool(tf.reduce_any(alive).numpy()):
                    time_step = next_time_step
                    break

                time_step = next_time_step

            # Episode metrics
            if last_slate is None:
                last_slate = np.zeros((num_users, slate_size), dtype=np.int32)
                last_choice = np.full((num_users,), slate_size, dtype=np.int32)
                last_reward = np.zeros((num_users,), dtype=np.float32)

            # (clicked-only) relevance from sentinel-safe choice
            relevance = np.zeros_like(last_slate, dtype=np.float32)
            click_mask = (last_choice >= 0) & (last_choice < slate_size)
            if np.any(click_mask):
                relevance[np.where(click_mask)[0], last_choice[click_mask]] = 1.0

            ndcg = ndcg_at_k(last_slate, relevance, k=slate_size)
            mrr = slate_mrr(last_slate, relevance, k=slate_size)
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            click_rate = float(np.mean(click_mask))
            eps_out = None
            if hasattr(agent.collect_policy, "epsilon"):
                try:
                    eps_out = float(agent.collect_policy.epsilon)
                except Exception:
                    eps_out = None

            log_payload = {
                "episode": int(episode),
                "total_reward": float(episode_reward),
                "loss": float(avg_loss),
                "ndcg@5": float(ndcg),
                "slate_mrr": float(mrr),
                "click_rate": float(click_rate),
            }
            if eps_out is not None:
                log_payload["epsilon"] = eps_out

            logger.log(log_payload)
            if eps_out is not None:
                print(f"[Episode {episode}] Total Reward: {episode_reward:.2f} | click_rate={click_rate:.3f} | ε={eps_out:.3f}")
            else:
                print(f"[Episode {episode}] Total Reward: {episode_reward:.2f} | click_rate={click_rate:.3f}")

            # episodic cleanup to keep memory steady
            del episode_losses, last_slate, last_reward, last_choice
            if episode % 20 == 0:
                gc.collect()

            if STOP:
                print("Stop requested. Exiting after episode cleanup...")
                break

        if FLAGS.peek_final_state:
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

    finally:
        logger.close()

        # Plotting
        if FLAGS.plot:
            try:
                metrics_df = pd.read_csv(logger.csv_file)

                # Dtypes & cleaning
                if "episode" not in metrics_df.columns:
                    raise ValueError("No 'episode' column found in metrics CSV.")
                metrics_df["episode"] = pd.to_numeric(metrics_df["episode"], errors='coerce')
                metrics_df = metrics_df.dropna(subset=["episode"])
                metrics_df["episode"] = metrics_df["episode"].astype(int)

                numeric_cols = ["total_reward", "loss", "ndcg@5", "slate_mrr", "click_rate", "epsilon"]
                for col in numeric_cols:
                    if col in metrics_df.columns:
                        metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')

                # Path handling
                log_dir_path = logger.log_dir if isinstance(logger.log_dir, Path) else Path(logger.log_dir)
                run_name = log_dir_path.name
                plots_dir = Path("plots") / agent_name
                plots_dir.mkdir(parents=True, exist_ok=True)

                plt.style.use("default")

                # X ticks every 100 episodes (ints only)
                ep_series = metrics_df["episode"]
                xticks = ep_series[ep_series % 100 == 0].astype(int).tolist()

                # helper: plot with mean line + moving average
                def plot_with_avg(df, ycol, title, ylabel, filename, ma_window=20):
                    if ycol not in df.columns or not df[ycol].notnull().any():
                        return
                    x = df["episode"].values
                    y = df[ycol].values
                    y_valid = y[~np.isnan(y)]
                    if y_valid.size == 0:
                        return
                    y_mean = float(np.mean(y_valid))
                    # moving average (trailing window)
                    y_ma = pd.Series(y).rolling(ma_window, min_periods=1).mean().values

                    plt.figure()
                    plt.plot(x, y, label=ycol)
                    plt.plot(x, y_ma, label=f"MA({ma_window})", linestyle='-.', linewidth=1.25)
                    plt.axhline(y=y_mean, linestyle='--', alpha=0.7, label=f"Mean = {y_mean:.2f}")
                    plt.xlabel("Episode")
                    plt.ylabel(ylabel)
                    plt.title(title)
                    if xticks:
                        plt.xticks(xticks)
                    plt.grid(True, linestyle='--', alpha=0.4)
                    plt.legend()
                    plt.savefig(plots_dir / f"{run_name}_{filename}.png")
                    plt.close()

                # Reward
                plot_with_avg(metrics_df, "total_reward",
                              "Total Reward over Episodes", "Total Reward", "reward")

                # Loss
                plot_with_avg(metrics_df, "loss",
                              "Training Loss over Episodes", "Loss", "loss")

                # NDCG and MRR
                has_ndcg = "ndcg@5" in metrics_df and metrics_df["ndcg@5"].notnull().any()
                has_mrr = "slate_mrr" in metrics_df and metrics_df["slate_mrr"].notnull().any()
                if has_ndcg or has_mrr:
                    plt.figure()
                    x = metrics_df["episode"].values

                    if has_ndcg:
                        y = metrics_df["ndcg@5"].values
                        y_ma = pd.Series(y).rolling(20, min_periods=1).mean().values
                        y_mean = float(np.nanmean(y))
                        plt.plot(x, y, label="NDCG@5")
                        plt.plot(x, y_ma, label="NDCG@5 MA(20)", linestyle='-.', linewidth=1.25)
                        plt.axhline(y=y_mean, linestyle='--', alpha=0.7, label=f"NDCG mean={y_mean:.3f}")

                    if has_mrr:
                        y2 = metrics_df["slate_mrr"].values
                        y2_ma = pd.Series(y2).rolling(20, min_periods=1).mean().values
                        y2_mean = float(np.nanmean(y2))
                        plt.plot(x, y2, label="Slate MRR")
                        plt.plot(x, y2_ma, label="MRR MA(20)", linestyle='-.', linewidth=1.25)
                        plt.axhline(y=y2_mean, linestyle='--', alpha=0.7, label=f"MRR mean={y2_mean:.3f}")

                    plt.xlabel("Episode"); plt.ylabel("Ranking Score")
                    plt.title("Ranking Metrics over Episodes")
                    if xticks:
                        plt.xticks(xticks)
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.savefig(plots_dir / f"{run_name}_ranking.png"); plt.close()

                # Click rate / epsilon
                fig_needed = ("click_rate" in metrics_df and metrics_df["click_rate"].notnull().any()) or \
                             ("epsilon" in metrics_df and metrics_df["epsilon"].notnull().any())
                if fig_needed:
                    plt.figure()
                    x = metrics_df["episode"].values

                    if "click_rate" in metrics_df and metrics_df["click_rate"].notnull().any():
                        y = metrics_df["click_rate"].values
                        y_ma = pd.Series(y).rolling(20, min_periods=1).mean().values
                        y_mean = float(np.nanmean(y))
                        plt.plot(x, y, label="Click Rate")
                        plt.plot(x, y_ma, label="Click MA(20)", linestyle='-.', linewidth=1.25)
                        plt.axhline(y=y_mean, linestyle='--', alpha=0.7, label=f"Click mean={y_mean:.3f}")

                    if "epsilon" in metrics_df and metrics_df["epsilon"].notnull().any():
                        y2 = metrics_df["epsilon"].values
                        y2_ma = pd.Series(y2).rolling(20, min_periods=1).mean().values
                        y2_mean = float(np.nanmean(y2))
                        plt.plot(x, y2, label="Epsilon")
                        plt.plot(x, y2_ma, label="Eps MA(20)", linestyle='-.', linewidth=1.25)
                        plt.axhline(y=y2_mean, linestyle='--', alpha=0.7, label=f"Eps mean={y2_mean:.3f}")

                    plt.xlabel("Episode"); plt.ylabel("Value")
                    plt.title("Click Rate (and Epsilon) over Episodes")
                    if xticks:
                        plt.xticks(xticks)
                    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
                    plt.savefig(plots_dir / f"{run_name}_click_epsilon.png"); plt.close()

                plt.close('all')

            except Exception as e:
                print(f"(Plotting skipped due to: {e})")

if __name__ == '__main__':
    app.run(main)
