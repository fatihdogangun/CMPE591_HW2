import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm  

from homework2 import Hw2Env

N_ACTIONS = 8
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.995 
EPSILON_DECAY_ITER = 1
MIN_EPSILON = 0.05
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
UPDATE_FREQ = 10          
TARGET_NETWORK_UPDATE_FREQ = 200  
BUFFER_LENGTH = 100000
MAX_EPISODES = 10000        

class Network(nn.Module):
    def __init__(self, input_dim=6, output_dim=N_ACTIONS):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        if next_state is None:
            next_state = np.zeros_like(state)
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = map(np.array, zip(*batch))
        return states, actions, rewards, next_states
    
    def __len__(self):
        return len(self.buffer)

def select_action(state, epsilon, q_network):
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(state_tensor)
        return q_values.argmax().item()

def main():
    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
    replay_buffer = ReplayBuffer(BUFFER_LENGTH)
    
    policy_net = Network()
    target_net =Network()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    
    epsilon = EPSILON_START
    global_step = 0

    episode_rewards = []
    episode_rps = []

    for episode in tqdm(range(MAX_EPISODES), desc="Training Episodes"):
        env.reset()
     
        state = env.high_level_state()
        cumulative_reward = 0.0
        episode_steps = 0
        done = False
        
        while not done:
            global_step += 1
            episode_steps += 1
            
           
            action = select_action(state, epsilon, policy_net)
            
            _, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated

            cumulative_reward += reward

            next_state = env.high_level_state()
            
            if done:
                next_state = None
                replay_buffer.push(state, action, reward, next_state)
            else: 
                replay_buffer.push(state, action, reward, next_state)
            
            state = next_state

            if len(replay_buffer) >= BATCH_SIZE and global_step % UPDATE_FREQ == 0:
                states, actions, rewards, next_states = replay_buffer.sample(BATCH_SIZE)
                
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
     
                current_q = policy_net(states).gather(1, actions)

                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
  
                if next_state is np.zeros_like(state):
                    target_q = rewards
                else:
                    target_q = rewards + GAMMA * max_next_q
                
                loss = nn.SmoothL1Loss()(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                
                
                optimizer.step()
                
            if global_step % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        rps = cumulative_reward / episode_steps if episode_steps > 0 else 0
        episode_rewards.append(cumulative_reward)
        episode_rps.append(rps)
        tqdm.write(f"Episode {episode+1}/{MAX_EPISODES}: Total Reward = {cumulative_reward:.2f}, RPS = {rps:.4f}, Epsilon = {epsilon:.3f}")

    torch.save(policy_net.state_dict(), "save/policy_net.pth")
    torch.save(target_net.state_dict(), "save/target_net.pth")

    np.save("save/episode_rewards.npy", np.array(episode_rewards))
    np.save("save/episode_rps.npy", np.array(episode_rps))

    smoothed_rewards = np.convolve(episode_rewards, np.ones(400)/400, mode="valid")
    smoothed_rps = np.convolve(episode_rps, np.ones(400)/400, mode="valid")

    plt.figure(figsize=(12, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode")
    plt.tight_layout()
    plt.grid()
    plt.savefig("episode_rewards.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(episode_rps)
    plt.xlabel("Episode")
    plt.ylabel("Reward per Step (RPS)")
    plt.title("RPS per Episode")
    plt.tight_layout()
    plt.grid()
    plt.savefig("episode_rps.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(smoothed_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Smoothed Reward per Episode")
    plt.tight_layout()
    plt.grid()
    plt.savefig("smoothed_episode_rewards.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(smoothed_rps)
    plt.xlabel("Episode")
    plt.ylabel("Reward per Step (RPS)")
    plt.title("Smoothed RPS per Episode")
    plt.tight_layout()
    plt.grid()
    plt.savefig("smoothed_episode_rps.png")
    plt.close()

if __name__ == '__main__':
   main()
