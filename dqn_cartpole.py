import math
import random

import matplotlib.pyplot as plt
import matplotlib

from collections import deque, namedtuple
from itertools import count

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 环境
env = gym.make('CartPole-v1')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else 
#     "mps" if torch.backends.mps.is_available() else
#     "cpu"
# )

device = torch.device("cpu")




# 经验回放
# 轨迹
Transition = namedtuple('Transition',
                         ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):   #capacity 是Replay Memory最大容量
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


# Q-Network
class DQN(nn.Module):
    
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)



# 超参数
BATCH_SIZE = 128    # 批量大小
GAMMA = 0.99    # 折扣因子
EPS_START = 0.9    # 探索开始时的epsilon
EPS_END = 0.05    # 探索结束时的epsilon
EPS_DECAY = 1000    # epsilon衰减的步数
TAU = 0.005    # 目标网络更新参数，用于 软更新target network
LR = 1e-4    # 学习率

# 取出动作空间和状态空间维度
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)


policy_net = DQN(n_observations, n_actions).to(device)    # 策略网络
target_net = DQN(n_observations, n_actions).to(device)    # 目标网络
target_net.load_state_dict(policy_net.state_dict())    # 目标网络初始化

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

memory = ReplayMemory(10000)


steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]],dtype=torch.long)


episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1)
        means = means.view(-1)
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    # 训练需要一个批量（batch），经验池没攒够，就先别学
    if len(memory) < BATCH_SIZE:
        return

    # 从经验池里随机拿一批
    transitions = memory.sample(BATCH_SIZE)

    # 用zip方法，将其分成states、actions、rewards、next_states的字段列表
    # 可以得到batch.state、 batch.action、 batch.next_state、 batch.reward
    batch = Transition(*zip(*transitions))


    # 对batch.next_state处理

    # 假设 batch.next_state = [s1, s2, None, s4]，将其转换成[True, True, False, True]
    # 最后转成 tensor bool类型，    non_final_mask是索引
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), 
        dtype=torch.bool)

    # 把[s1, s2, None, s4] 变成：[s1, s2, s4]； 用于后续只对“还没结束的状态”计算 Q(s′)
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )

    # 构造训练用 batch Tensor
    state_batch = torch.cat(batch.state)    # 状态批量
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 策略网络计算预测Q值
    # policy_net(state_batch)最终输出[BATCH_SIZE, n_actions]形式的Q值，即一个状态两个Q值
    # 计算当前 Q(s,a)，其中gather方法是只取“当初实际执行的那个动作”的Q值
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 目标网络计算TD目标值，计算目标 Q（target）
    next_state_values = torch.zeros(BATCH_SIZE, device=device) #初始化为0
    with torch.no_grad():      # 不记录梯度，target_net不参与反向传播
        # 计算 max Q(s′, a′)， non_final_mask是batch.next_state的索引
        next_state_values[non_final_mask] = target_net(
            non_final_next_states
        ).max(1)[0]     # .max(1) → 沿action维度；  [0] → 只要max值，不要index

    # 最终的 TD Target
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 损失计算
    criterion = nn.SmoothL1Loss() 
    loss = criterion(
        state_action_values,    #策略网络保留的实际执行动作的Q值
        expected_state_action_values.unsqueeze(1) #目标网络计算的TD目标，还需要补充一个维度对齐
    )
    
    # 反向传播 and 梯度裁剪
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()




# 每个 episode：从reset开始，到杆子倒下结束
# 一次 episode = 一整条轨迹
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 500

for i_episode in range(num_episodes):
    state, info = env.reset()

    state = torch.tensor(
        state, dtype=torch.float32, device=device
    ).unsqueeze(0)

    for t in count():
        action = select_action(state) # 选动作，连接到 ε-greedy
        
        # 把动作交给环境，  .item() → Python int；  Gym环境不接受tensor
        # terminated是否自然终止；truncated是否时间截断
        next_state, reward, terminated, truncated, info = env.step(action.item())
        
        # 
        reward = torch.tensor([reward], device=device)
        
        # 处理next_state， 终止状态没有next_state
        done = terminated or truncated

        if done:
            next_state = None
        else:
            next_state = torch.tensor(
                next_state, dtype=torch.float32, device=device
            ).unsqueeze(0)

        # 把经验存进 Replay Memory
        memory.push(state, action, next_state, reward)

        state = next_state #状态推进

        optimize_model() #优化模型，真正学习发生在这里

        # Target Network 的软更新
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key].data.copy_(policy_net_state_dict[key].data)#硬更新
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()