import os
import toml
from copy import deepcopy
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import system as s

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
        """
        経験再生バッファの初期化
        Args:
        - buffer_size (int): バッファに保存できるデータの最大数
        - batch_size (int): バッチごとに取得するデータの数
        """
        self.buffer = deque(maxlen=buffer_size)  # 経験データを格納するデータ構造
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        バッファに新しい経験データを追加
        Args:
        - state (np.ndarray): 現在の状態
        - action (int): 実行した行動
        - reward (float): 行動による報酬
        - next_state (np.ndarray): 次の状態
        - done (bool): エピソード終了フラグ
        """
        data = (
            torch.tensor(state, dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.int64, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(next_state, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device)
        )
        self.buffer.append(data)  # 新しい経験をバッファに追加

    def __len__(self):
        """
        バッファに格納されている経験の数を返す
        """
        return len(self.buffer)

    def get_batch(self):
        """
        バッファからランダムにバッチサイズ分の経験を取得
        Returns:
        - state (torch.Tensor): バッチの状態
        - action (torch.Tensor): バッチの行動
        - reward (torch.Tensor): バッチの報酬
        - next_state (torch.Tensor): バッチの次の状態
        - done (torch.Tensor): バッチのエピソード終了フラグ
        """
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
    def __init__(self, state_size:int, action_size:int):
        """
        Qネットワークの初期化
        4層の全結合層を持つネットワーク
        Args:
        - action_size (int): 行動のサイズ（行動の選択肢数）
        """
        super().__init__()
        self.l1 = nn.Linear(state_size, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, action_size)

    def forward(self, x):
        """
        ネットワークのフォワードパス
        Args:
        - x (torch.Tensor): 入力テンソル（状態）
        Returns:
        - torch.Tensor: 行動価値の予測結果
        """
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
        """
        DQNエージェントの初期化
        強化学習に必要なパラメータを設定し、Qネットワークを初期化
        """
        self.epsilon = 0.1
        self.gamma = 0.999
        self.lr = 0.0005
        self.buffer_size = 10000
        self.batch_size = 128
        self.state_size = 3
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

    def init_nn(self):
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.state_size, self.action_size).to(device)
        self.qnet_target = QNet(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def getConfig(self):
        data = self.system.getConfig()

        data["simulation"]["current"] = self.current

        data["hyperparameters"] = {
                "epsilon": self.epsilon,
                "gamma": self.gamma,
                "lr": self.lr
            }

        data["training"] = {
                "episodes": self.episodes,
                "da": self.da,
                "a_step": self.a_step
            }
        return data

    def output(self, file:str = "config.toml"):
        data = self.getConfig()
        with open(file, 'w') as f:
            toml.dump(data, f)

    def sync_qnet(self):
        """
        Qネットワークのパラメータをターゲットネットワークにコピー
        """
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def get_action(self, state, epsilon):
        """
        行動選択
        ε-greedy法に基づき、ランダムまたはQネットワークの予測に基づいて行動を選択
        Args:
        - state (np.ndarray): 現在の状態
        - epsilon (float): 探索率（ランダム行動を選ぶ確率）
        Returns:
        - int: 選択された行動
        """
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            state_t = torch.tensor(state[np.newaxis, :]).to(device)
            return self.qnet(state_t).argmax().item()

    def update(self, state, action, reward, next_state, done):
        """
        経験再生バッファにデータを追加し、Qネットワークの重みを更新
        Args:
        - state (np.ndarray): 現在の状態
        - action (int): 実行した行動
        - reward (float): 得た報酬
        - next_state (np.ndarray): 次の状態
        - done (bool): エピソード終了フラグ
        Returns:
        - loss.data (torch.Tensor): 損失関数
        """
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

    def set_state(self, i):
        state = self.system.m[i]
        return state

    def set_current(self, old_current, action):
        return self.current * int(action != 0)

    def perform(self, echo=True, save=True):
        start_time = datetime.now()

        self.init_nn()

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
            i = 0
            state = self.set_state(i)
            action = self.get_action(state, self.epsilon)
            current = self.set_current(0e0, action)

            for i in range(self.system.steps):
                self.system.RungeKutta(current)

                if i > 0 and i % self.a_step == 0:
                    reward = - self.system.m[i,0]**3
                    self.reward_history[episode] += reward
                    next_state = self.set_state(i)
                    next_action = self.get_action(next_state, self.epsilon)

                    loss = self.update(state, action, reward, next_state, done)

                    state = next_state
                    action = next_action

                current = self.set_current(current, action)

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
                last_action = action
                for _ in range(self.post_eval_steps):
                    post_sys.RungeKutta(0e0)
                next_penalty = - post_sys.m[-1,0]**3 * self.post_penalty_factor
                fluctuation = np.std(post_sys.m[:,0])
                fluctuation_penalty = fluctuation * fluctuation_penalty_factor
                post_penalty = next_penalty + fluctuation_penalty
                last_state = self.set_state(-1)
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
        self.output(self.directory+"config.toml")
        self.system.m = self.best_m
        self.system.j = self.best_j
        np.savetxt(self.directory+"t.txt", self.system.t)
        label = "best"
        self.system.save_data(label, self.directory)
        self.system.save_episode(label, self.directory)
        self.save_reward_history()

class ExtendedDQNAgent(DQNAgent):
    def __init__(self, episodes:int, record:int, sync_interval:int,
                 system:s.System, current:np.float64, da:np.float64,
                 post_eval_time:np.float64 = 0.5e-9, post_penalty_factor:np.float64 = 1.0,
                 fluctuation_penalty_factor:np.float64 = 1.0,
                 directory:str = ""):
        super().__init__(episodes, record, sync_interval, system, current, da,
                         post_eval_time, post_penalty_factor,
                         fluctuation_penalty_factor,
                         directory)
        self.state_size = 4
        self.action_size = 3

    def set_state(self, i):
        state = np.append(self.system.m[i], [self.system.j[i]])  # 状態を連結して作成
        return state

    def set_current(self, old_current, action):
        j0 = action - 1
        current = old_current + self.current*j0*self.system.dt/self.da  # 電流密度を更新
        return current

if __name__ == '__main__':
    T = 0 # 温度 [K]
    M = 750e3  # 飽和磁化　[A/m]

    end = 0.8e-9  # シミュレーションの終了時間 [秒]
    dt = 1e-12  # タイムステップ [秒]
    alphaG = 0.01e0  # ギルバート減衰定数
    beta = -3e0  # field like torque と damping like torque の比
    theta = -0.25e0  # スピンホール角
    size = np.array([100e-9, 50e-9, 1e-9]) # [m] 強磁性体の寸法
    d_Pt = 5.0e-9  # Ptの厚み [m]
    H_appl = np.array([0e0, 0e0, 0e0]) # 外部磁場 [T]
    H_ani = np.array([0e0, 0e0, 0e0])  # 異方性定数 [T]
    m0 = np.array([1e0, 0e0, 0e0])     # 初期磁化

    post_eval_time = 0.5e-9
    post_penalty_factor = 1.0,
    fluctuation_penalty_factor = 1.0

    sys = s.ThermalSystem(end, dt, alphaG, beta, theta, size, d_Pt, M, H_appl, H_ani, m0, T)

    record = 10  # 結果の記録間隔
    sync_interval = 20  #　ターゲットネットワークを同期する間隔
    da = 20e-12  # 行動間隔 [秒]

#    episodes = 100  # エピソード数
#    current = 10e0 # 電流密度 [MA/cm2]
#    directory = "./"
#    agent = DQNAgent(episodes, record, sync_interval, sys, current, da, post_eval_time, post_penalty_factor, fluctuation_penalty_factor, directory)
#    agent.perform()
#    agent.save()

    episodes = 1000  # エピソード数
    current = 1e0 # 電流密度 [MA/cm2]
    directory = "extended/"
    agent = ExtendedDQNAgent(episodes, record, sync_interval, sys, current, da, post_eval_time, post_penalty_factor, fluctuation_penalty_factor, directory)
    agent.perform()
    agent.save()
