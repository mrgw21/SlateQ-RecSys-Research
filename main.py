import gin
from absl import app
from absl import flags
from recsim_ng.core import network as tf_network
from src.runtimes.ecomm_runtime import ECommRuntime
from src.stories import ecomm_story
from src.agents.slateq_agent import SlateQAgent
from recsim_ng.core import value
from tf_agents.replay_buffers import tf_uniform_replay_buffer

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('gin_files', [], 'Paths to the config files.')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin parameter bindings.')

def main(argv):
    del argv  # Unused.
    gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)
    story = ecomm_story(num_users=10, num_items=100, slate_size=5)
    network = tf_network.Network(variables=story)  # Updated class name
    rt = ECommRuntime(network=network)

    # Define action spec for slate (10 users, 5 items each)
    action_spec = value.ValueSpec(slate=value.FieldSpec(shape=(10, 5), dtype=tf.int32))

    # Initialize SlateQAgent
    agent = SlateQAgent(time_step_spec=rt.time_step_spec(), action_spec=action_spec, num_users=10, num_topics=10, slate_size=5)

    # Replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=1,
        max_length=10000
    )
    dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=32, num_steps=2).prefetch(3)

    # Training loop
    num_episodes = 10
    for episode in range(num_episodes):
        time_step = rt.reset()
        episode_reward = 0.0
        for step in range(5000):
            action_step = agent.policy.action(time_step)
            next_time_step = rt.step(action_step.action)
            experience = tf_agents.trajectories.trajectory.from_transition(time_step, action_step, next_time_step)
            replay_buffer.add_batch(experience)
            episode_reward += next_time_step.reward.numpy().sum()
            time_step = next_time_step

            # Train with batch
            for batch in dataset.take(1):
                loss_info = agent.train(batch)
                if step % 1000 == 0:
                    print(f"Episode {episode}, Step {step}, Loss: {loss_info.loss.numpy()}")

        print(f"Episode {episode} completed, Total Reward: {episode_reward}")

    # Retrieve and print final trajectory
    trajectory = rt.current_value()
    user_state = trajectory.get('user_state')
    if user_state:
        interest = user_state.get('interest')
        if isinstance(interest, tf.Tensor):
            for i, value in enumerate(interest.numpy()):
                print(f"User {i} final interest: {value}")
        else:
            print(f"Interest: {interest}")
    else:
        print("No user_state in trajectory")

if __name__ == '__main__':
    app.run(main)