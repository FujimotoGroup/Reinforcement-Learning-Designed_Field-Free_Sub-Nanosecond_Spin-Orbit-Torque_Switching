import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 18
import os
import re
import glob

from modules import system as s

root_dir = "./data/thermal/"

def main():
    directory = "./data/"
    print(directory)
    LLG(directory)

def LLG(directory):
    T = 300 # [K]

    # load files
    config = toml.load(directory+"config.toml")
    with open(directory+'/j_best.txt', 'r') as file:
        content = file.read()
    j_read = [float(val) for val in re.findall(r'-?\d+\.\d+e[+-]?\d+', content)]

    # シミュレーション設定
    end = 2.0e-9 # シミュレーションの終了時間 [秒]
    dt = config["simulation"]["dt"] # タイムステップ [秒]
    steps = int(end / dt)  # ステップ数
    alphaG = config["material"]["alphaG"]  # ギルバート減衰定数
    beta = config["material"]["beta"]  # field like torque と damping like torque の比
    theta = config["material"]["theta"]  # スピンホール角
    size = np.array(config["geometry"]["size"]) # 強磁性体の寸法 [m]
    V = size[0]*size[1]*size[2]  # 強磁性体の体積 [m^3]
    d_Pt = config["material"]["d_Pt"]  # Ptの厚み [m]
    M = config["material"]["M"] # 飽和磁化 [emu/c.c. = 10^3 A/m]
    H_appl = np.array(config["fields"]["H_appl"]) # 外部磁場 [T]
    H_ani = np.array(config["fields"]["H_ani"]) # 異方性定数 [T]
    H_shape =np.array(config["fields"]["H_shape"] # 反磁場 [T]
    j = np.zeros(steps+1, dtype=np.float64)
    j[:len(j_read)] = j_read

    m0 = np.array(config["simulation"]["m0"])  # 初期磁化
    current = config["simulation"]["current"] # 印加電流密度

    save_dir = root_dir+f"M{M*1e-3:.0f}/J{current:04.1f}e10_T{T:.0f}/"   # 結果を保存するディレクトリ名

    os.mkdir(directory)  # ディレクトリを作成

    # シミュレーション実行
    for n in range(1, 1001):
        print(n)
        system = s.ThermalSystem(end, dt, alphaG, beta, theta, size, d_Pt, M, H_appl, H_ani, m0, T)
        system.j = j

        system.run()

        # 結果を保存
        label = f"{n:03d}"
        system.save_episode(label, save_dir)
        np.savetxt(save_dir+label+".txt", system.m)
        system.output(save_dir+"config.toml")

if __name__ == '__main__':
    main()
