import numpy as np
import gym

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array")

alpha = 0.8
gamma = 0.95
epsilon = 0.1
num_episodes = 10000  

Q = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state, epsilon):
    """Choose an action using the epsilon-greedy policy."""
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample() 
    else:
        return np.argmax(Q[state, :])

for episode in range(num_episodes):
    state = env.reset()[0]
    done = False

    while not done:
        action = choose_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        state = next_state

print("Training complete! Q-table values:")
print(Q)

state = env.reset()[0]
done = False

env.render()
while not done:
    action = np.argmax(Q[state, :])
    state, reward, done, _, _ = env.step(action)
    env.render()

print(f"Test completed with reward: {reward}")