import gym
import numpy as np
env = gym.make("FrozenLake-v1")

alpha = 0.4
gamma = 0.999

q_table = dict([(x, [1, 1, 1, 1]) for x in range(16)])
score = []

def choose_action(observation):
    return np.argmax(q_table[observation])

for i in range(1000):
    observation = env.reset()
    env.render()
    action = choose_action(observation)

    prev_observation = None
    prev_action      = None

    t = 0

    for t in range(250):
        observation, reward, done, info = env.step(action)
        env.render()
        action = choose_action(observation)

        if not prev_observation is None:
            q_old = q_table[prev_observation][prev_action]
            q_new = q_old
            if done:
                q_new += alpha * (reward - q_old)
            else:
                q_new += alpha * (reward + gamma * q_table[observation][action] - q_old)

            new_table = q_table[prev_observation]
            new_table[prev_action] = q_new
            q_table[prev_observation] = new_table

        prev_observation = observation
        prev_action = action

        if done:
            if len(score) < 100:
                score.append(reward)
            else:
                score[i % 100] = reward

            print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(i, t, reward, np.mean(score)))
            break