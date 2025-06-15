import numpy as np
import os

from modules import system as s
from modules import agent  as a

T = 300 # 温度 [K]
end = 0.8e-9  # シミュレーションの終了時間 [秒]
dt = 1e-12  # タイムステップ [秒]
alphaG = 0.010  # ギルバート減衰定数
beta = -3  # field like torque と damping like torque の比
theta = -0.25  # スピンホール角
size = np.array([100e-9, 50e-9, 1e-9]) # [m] 強磁性体の寸法
#size = np.array([80e-9, 25e-9, 1e-9]) # [m] 強磁性体の寸法
d_Pt = 5.0e-9  # Ptの厚み [m]
H_appl = np.array([0e0, 0e0, 0e0]) # 外部磁場 [T]
H_ani = np.array([0e0, 0e0, 0e0])  # 異方性定数 [T]
m0 = np.array([1e0, 0e0, 0e0])     # 初期磁化

episodes = 2000  # エピソード数
record = 10  # 結果の記録間隔
sync_interval = 20  #　ターゲットネットワークを同期する間隔
da = 20e-12  # 行動間隔 [秒]

directory = f"../data/finiteT/{size[0]*1e9:.0f}x{size[1]*1e9:.0f}x{size[2]*1e9:.0f}/aG{alphaG:.3f}/"   # 結果を保存するディレクトリ名

def run_agent(M, J):
    post_eval_time = 0.5e-9
    post_penalty_factor = 2.0e0
    fluctuation_penalty_factor = 1e0
    current = J
    sys = s.ThermalSystem(end, dt, alphaG, beta, theta, size, d_Pt, M, H_appl, H_ani, m0, T)
    agent = a.DQNAgent(episodes, record, sync_interval, sys, current, da, post_eval_time, post_penalty_factor, fluctuation_penalty_factor, directory)
#    comment = agent.perform(echo=False,save=False)
#    comment = agent.perform(echo=False,save=True)
    comment = agent.perform(echo=True,save=True)
    agent.save()

    end_judge = 2.0e-9
    rule_period = 1.5e-9
    rule_range = -0.85e0
    sys.set(end_judge)

    res = sys.judge(agent.best_j, rule_period, rule_range)
    label = "judge"
    np.savetxt(agent.directory+"t_judge.txt", sys.t)
    sys.save_data(label, agent.directory)
    sys.save_episode(label, agent.directory)

    return res, comment

if __name__ == '__main__':
    print(directory)
    os.makedirs(directory, exist_ok=True)   # 結果を保存するディレクトリ

    M = 750e3
    J = 3e0
    run_agent(M,J)

    M = 750e3
    J = 5e0
    run_agent(M,J)

    M = 750e3
    J = 10e0
    run_agent(M,J)
