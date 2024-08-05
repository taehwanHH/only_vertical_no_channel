import torch
import os
import numpy as np
from DDPG import DDPG, Actor, Critic, prepare_training_inputs
from DDPG import OrnsteinUhlenbeckProcess as OUProcess

from memory import ReplayMemory
from train_utils import to_tensor
from target_update import soft_update
from Archery_env.Archery import ArcheryEnv
import matplotlib.pyplot as plt
from scipy.io import savemat
from collections import deque
from param import Hyper_Param
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'

# Hyperparameters
DEVICE = Hyper_Param['DEVICE']
tau = Hyper_Param['tau']
lr_actor = Hyper_Param['lr_actor']
lr_critic = Hyper_Param['lr_critic']
batch_size = Hyper_Param['batch_size']
gamma = Hyper_Param['discount_factor']
memory_size = Hyper_Param['memory_size']
total_eps = Hyper_Param['num_episode']
sampling_only_until = Hyper_Param['train_start']
print_every = Hyper_Param['print_every']
window_size = Hyper_Param['window_size']
step_max = Hyper_Param['step_max']

# List storing the results
score_avg = deque(maxlen=window_size)
cum_score_list = []
score_avg_value = []
epi = []

# Create Environment
env = ArcheryEnv()
s_dim = env.state_dim
a_dim = env.action_dim

# initialize target network same as the main network.
actor, actor_target = Actor(), Actor()
critic, critic_target = Critic(), Critic()

agent = DDPG(critic=critic,
             critic_target=critic_target,
             actor=actor,
             actor_target=actor_target,epsilon= Hyper_Param['epsilon'],
             lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma).to(DEVICE)

memory = ReplayMemory(memory_size)

# Episode start
for n_epi in range(total_eps):
    ou_noise = OUProcess(mu=np.zeros(2))
    s = env.reset()
    epi.append(n_epi)

    while True:
        s = to_tensor(s, size=(-1, s_dim))
        a = agent.get_action(s, agent.epsilon*ou_noise()[:2])
        ns, r, done, info = env.step(a)
        experience = (s,
                      a.view(-1,a_dim),
                      torch.tensor(r).view(-1, 1),
                      torch.tensor(ns).view(-1, s_dim),
                      torch.tensor(done).view(-1, 1))
        memory.push(experience)
        env.state = ns
        s = env.state
        if done:
            break

    cum_score = env.cum_score/step_max
    score_avg.append(cum_score)
    cum_score_list.append(cum_score)

    if len(memory) >= sampling_only_until:
        # train agent
        agent.epsilon = max(agent.epsilon * Hyper_Param['epsilon_decay'], Hyper_Param['epsilon_min'])

        sampled_exps = memory.sample(batch_size)
        sampled_exps = prepare_training_inputs(sampled_exps)
        agent.update(*sampled_exps)

        soft_update(agent.actor, agent.actor_target, tau)
        soft_update(agent.critic, agent.critic_target, tau)


    if len(score_avg) == window_size:
        score_avg_value.append(sum(score_avg) / window_size)

    else:
        score_avg_value.append(sum(score_avg) / len(score_avg))

    if n_epi % print_every == 0:
        msg = (n_epi, cum_score, agent.epsilon)
        print("Episode : {:4.0f} | Cumulative score : {:.2f} | epsilon : {:.3f}:".format(*msg))
        plt.xlim(0, total_eps)
        plt.ylim(0, 10)
        plt.plot(epi, cum_score_list, color='black')
        plt.plot(epi, score_avg_value, color='red')
        # plt.plot(epi, optimal_score_avg_value, color='blue')
        # plt.plot(epi, cum_rand_score_list, color='blue')
        # plt.plot(epi, cum_optimal_score_list, color='green')
        plt.xlabel('Episode', labelpad=5)
        plt.ylabel('Average score', labelpad=5)
        plt.grid(True)
        plt.pause(0.0001)
        plt.close()


# Base directory path creation
base_directory = os.path.join(Hyper_Param['today'])

# Subdirectory index calculation
if not os.path.exists(base_directory):
    os.makedirs(base_directory)
    index = 1
else:
    existing_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    indices = [int(d) for d in existing_dirs if d.isdigit()]
    index = max(indices) + 1 if indices else 1

# Subdirectory creation
sub_directory = os.path.join(base_directory, str(index))
os.makedirs(sub_directory)

# Store plt in Subdirectory
plt.xlim(0, total_eps)
plt.ylim(0, 10)
plt.plot(epi, cum_score_list, color='black')
plt.plot(epi, score_avg_value, color='red')
# plt.plot(epi, optimal_score_avg_value, color='blue')
# plt.plot(epi, cum_rand_score_list, color='blue')
# plt.plot(epi, cum_optimal_score_list, '--g')
plt.xlabel('Episode', labelpad=5)
plt.ylabel('Average score', labelpad=5)
plt.grid(True)
plt.savefig(os.path.join(sub_directory, f"plot_{index}.png"))

# Store Hyperparameters in txt file
with open(os.path.join(sub_directory, 'Hyper_Param.txt'), 'w') as file:
    for key, value in Hyper_Param.items():
        file.write(f"{key}: {value}\n")

# Store score data (matlab data file)
savemat(os.path.join(sub_directory, 'data.mat'),{'sim_res': cum_score_list})
# savemat(os.path.join(sub_directory, 'data.mat'),{'sim_res': cum_score_list,'sim_optimal': optimal_score_avg_value})
# savemat(os.path.join(sub_directory, 'data.mat'), {'sim_res': cum_score_list,'sim_rand_res': cum_rand_score_list,
#                                                   'sim_optimal_res': cum_optimal_score_list})
