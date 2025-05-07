import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

import numpy as np
import os

from modules import system as s
from modules import agent  as a

T = 0 # 温度 [K]
end = 0.8e-9  # シミュレーションの終了時間 [秒]
dt = 1e-12  # タイムステップ [秒]
alphaG = 0.01  # ギルバート減衰定数
beta = -3  # field like torque と damping like torque の比
theta = -0.25  # スピンホール角
#size = np.array([100e-9, 50e-9, 1e-9]) # [m] 強磁性体の寸法
size = np.array([80e-9, 25e-9, 1e-9]) # [m] 強磁性体の寸法
d_Pt = 5.0e-9  # Ptの厚み [m]
H_appl = np.array([0e0, 0e0, 0e0]) # 外部磁場 [T]
H_ani = np.array([0e0, 0e0, 0e0])  # 異方性定数 [T]
m0 = np.array([1e0, 0e0, 0e0])     # 初期磁化

episodes = 1000  # エピソード数
record = 10  # 結果の記録間隔
sync_interval = 20  #　ターゲットネットワークを同期する間隔
da = 20e-12  # 行動間隔 [秒]

directory = f"./data/{size[0]*1e9:.0f}x{size[1]*1e9:.0f}x{size[2]*1e9:.0f}/aG{alphaG:.3f}/"   # 結果を保存するディレクトリ名

def run_agent(M, J):
    post_eval_time = 0.5e-9
    post_penalty_factor = 2.0e0
    fluctuation_penalty_factor = 1e0
    current = J
    sys = s.ThermalSystem(end, dt, alphaG, beta, theta, size, d_Pt, M, H_appl, H_ani, m0, T)
    agent = a.DQNAgent(episodes, record, sync_interval, sys, current, da, post_eval_time, post_penalty_factor, fluctuation_penalty_factor, directory)
#    comment = agent.perform(echo=False,save=False)
    comment = agent.perform(echo=False,save=True)
#    comment = agent.perform(echo=True,save=True)
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
    multiprocessing.set_start_method('spawn')
    # 以下修正可能 ------------------------------------------------------------------------
    T = 0

    min_M = 500e3  # 最大飽和磁化 [A/m]
    max_M = 2500e3 # 最小飽和磁化 [A/m]
    dM    = 250e3  # 飽和磁化刻み幅 [A/m]

#    min_J = 5e9   # [MA/cm2]
#    max_J = 11e10 # [MA/cm2]
#    dJ    = 5e9   # [MA/cm2]

    min_J =  2e10   # [MA/cm2]
    max_J = 20e10 # [MA/cm2]
    dJ    =  2e10   # [MA/cm2]
    # 以上修正可能 ------------------------------------------------------------------------

    Ms = list(range(int(min_M), int(max_M) + int(dM), int(dM)))
    Js = list(range(int(min_J), int(max_J) + int(dJ), int(dJ)))

    print(directory)
    os.makedirs(directory, exist_ok=True)   # 結果を保存するディレクトリ

#    M = 500e3
#    J = 5e0
#    run_agent(M,J)
#    exit()

    data = []  # 結果保存リスト
    MJs = list(itertools.product(Ms, Js))

    with ProcessPoolExecutor(max_workers=15) as executor:
        data = []
        param = {
            executor.submit(run_agent, M, J*1e-10): (M, J) for M, J in MJs
        }
        total = len(param)

        for i, f in enumerate(as_completed(param), 1):
            M, J = param[f]
            result, comment = f.result()
            data.append([M, J, result])
            print(f"[{i:03d}/{total}] Completed: M={M:04d}, J={J*1e-10:04.1f}e10 "+comment)

    np.savetxt(directory+"/scatter_data.txt", data)
