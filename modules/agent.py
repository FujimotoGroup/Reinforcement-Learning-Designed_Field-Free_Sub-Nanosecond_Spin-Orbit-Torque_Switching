import os
from copy import deepcopy
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import system as s

from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.rcParams.update({
    "text.usetex": False,
})
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams["font.size"] = 20

device = torch.device("cuda:0")  # NVIDIA GPU (GPU 1) を指定

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (
            torch.tensor(state, dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.int64, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(next_state, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device)
        )
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*data)
        return (
            torch.stack(state),
            torch.tensor(action, dtype=torch.int64, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.stack(next_state),
            torch.tensor(done, dtype=torch.float32, device=device),
        )

class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(3, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, action_size)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)

class DQNAgent:
    def __init__(self, episodes:int, record:int, sync_interval:int,
                 system:s.System, current:np.float64, da:np.float64,
                 post_eval_time:np.float64 = 0.5e-9, post_penalty_factor:np.float64 = 1.0,
                 fluctuation_penalty_factor:np.float64 = 1.0,
                 directory:str = ""):
        self.epsilon = 0.1
        self.gamma = 0.999
        self.lr = 0.0005
        self.buffer_size = 10000
        self.batch_size = 128
        self.action_size = 2

        self.episodes = episodes
        self.total_loss = 0e0
        self.reward_history = np.zeros(episodes)
        self.best_reward = -500
        self.record = record
        self.system = system
        self.current = current
        self.da = da*1e9
        self.sync_interval = sync_interval
        self.a_step = int(self.da / self.system.dt)
        self.directory = directory+f"M{self.system.M*1e-3:.0f}/J{self.current:04.1f}e10_T{self.system.T:.0f}/"
        os.makedirs(self.directory, exist_ok=True)

        self.best_m = self.system.m
        self.best_j = self.system.j

        # ポストエピソード評価用設定
        self.post_eval_time = post_eval_time
        self.post_penalty_factor = post_penalty_factor
        self.post_eval_steps = int(self.post_eval_time / self.system.dt)
        self.fluctuation_penalty_factor = fluctuation_penalty_factor

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size).to(device)
        self.qnet_target = QNet(self.action_size).to(device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            state_t = torch.tensor(state[np.newaxis, :]).to(device)
            return self.qnet(state_t).argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return None
        state_b, action_b, reward_b, next_state_b, done_b = self.replay_buffer.get_batch()
        qs = self.qnet(state_b)
        q = qs[np.arange(self.batch_size), action_b]
        next_q = self.qnet_target(next_state_b).max(1)[0].detach()
#        target = reward_b + (1.0 - done_b) * self.gamma * next_q
        target = reward_b + self.gamma * next_q
        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def perform(self, echo=True, save=True):
        start_time = datetime.now()
        for episode in range(self.episodes):
            if episode < self.episodes/2:
                self.epsilon = 1.0 + (0.1 - 1.0)/(self.episodes/2)*(episode)
            elif episode > self.episodes*0.975:
                self.epsilon = 0
                self.record = 5
            else:
                self.epsilon = 0.1

            self.system.reset()
            done = 0
            old_state = self.system.m[0]
            action = self.get_action(old_state, self.epsilon)
            current = self.current if action != 0 else 0

            for i in range(self.system.steps):
                self.system.RungeKutta(current)
                if i > 0 and i % self.a_step == 0:
                    reward = - self.system.m[i,0]**3
                    self.reward_history[episode] += reward
                    loss = self.update(old_state, action, reward, self.system.m[i], done)
                    old_state = self.system.m[i]
                    action = self.get_action(old_state, self.epsilon)
                    current = self.current if action != 0 else 0
                if episode > self.sync_interval:
                    self.total_loss += loss if loss is not None else 0

            if episode % self.sync_interval == 0:
                self.sync_qnet()

            if save and episode % self.record == self.record-1:
                label = "episode{:0=5}".format(episode+1)
                self.system.save_data(label, self.directory)
                self.system.save_episode(label, self.directory)

            # ポストエピソード評価で最後の行動をペナルティ修正
            if self.post_eval_steps > 0:
                post_sys = deepcopy(self.system)
                post_sys.set(self.post_eval_time)
                post_sys.m[0] = self.system.m[-1]
                last_state = self.system.m[-1]
                last_action = action
                for _ in range(self.post_eval_steps):
                    post_sys.RungeKutta(0e0)
                next_penalty = - post_sys.m[-1,0]**3 * self.post_penalty_factor
                fluctuation = np.std(post_sys.m[:,0])
                fluctuation_penalty = fluctuation * fluctuation_penalty_factor
                post_penalty = next_penalty + fluctuation_penalty
                self.replay_buffer.add(last_state, last_action, post_penalty, post_sys.m[-1], True)
                self.reward_history[episode] += post_penalty

            # ベストエピソードの更新
            if episode > self.episodes*0.975:
                if self.reward_history[episode] > self.best_reward:
                    self.best_episode = episode + 1
                    self.best_reward = self.reward_history[episode]
                    self.best_m = self.system.m.copy()
                    self.best_j = self.system.j.copy()

            if echo:
                print(f"episode:{episode+1:>4}: reward = {self.reward_history[episode]:.9f}")

        end_time = datetime.now()
        duration = end_time - start_time
        h, rem = divmod(duration.total_seconds(), 3600)
        m, s = divmod(rem, 60)
        return f"start:{start_time.strftime('%Y/%m/%d %H:%M')}, end:{end_time.strftime('%H:%M')}, duration: {int(h)}h{int(m):02d}m{s:.1f}s"

    def save_reward_history(self):
        episodes = np.arange(self.episodes)

        slice_num = 20
        average = [ self.reward_history[i:i+slice_num].mean() for i in episodes[::slice_num]]

        plt.figure(figsize=(6,6))
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(episodes, self.reward_history, label='Rewards for each Episode')
        plt.plot(episodes[::slice_num], average, label='Average Rewards for 20 Episodes')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.directory+"reward_history.png", dpi=200)
        plt.close()

        np.savetxt(self.directory+"reward_history.txt", self.reward_history)

    def save(self):
        self.system.output(self.directory+"config.toml")
        self.system.m = self.best_m
        self.system.j = self.best_j
        np.savetxt(self.directory+"t.txt", self.system.t)
        label = "best"
        self.system.save_data(label, self.directory)
        self.system.save_episode(label, self.directory)
        self.save_reward_history()

if __name__ == '__main__':
    T = 300 # 温度 [K]
    M = 750e3  # 飽和磁化　[A/m]
    current = 10e10 # 電流密度 [A/m2]

    end = 0.8e-9  # シミュレーションの終了時間 [秒]
    dt = 1e-12  # タイムステップ [秒]
    alphaG = 0.05  # ギルバート減衰定数
    beta = -3  # field like torque と damping like torque の比
    theta = -0.25  # スピンホール角
    size = np.array([100e-9, 50e-9, 1e-9]) # [m] 強磁性体の寸法
    d_Pt = 5.0e-9  # Ptの厚み [m]
    H_appl = np.array([0e0, 0e0, 0e0]) # 外部磁場 [T]
    H_ani = np.array([0e0, 0e0, 0e0])  # 異方性定数 [T]
    m0 = np.array([1e0, 0e0, 0e0])     # 初期磁化

    sys = ThermalSystem(end, dt, alphaG, beta, theta, size, d_Pt, M, H_appl, H_ani, m0, T)

    episodes = 100  # エピソード数
    record = 10  # 結果の記録間隔
    sync_interval = 20  #　ターゲットネットワークを同期する間隔
    da = 20e-12  # 行動間隔 [秒]

    agent = DQNAgent(episodes, record, sync_interval, sys, current, da)  # DQNエージェントの初期化
    agent.perfom()
    agent.save()
