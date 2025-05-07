import toml
import numpy as np
import os
import re
import glob

from modules import system as s

load_directory = "./data/100x50x1/aG0.010/M750/J06.0e10_T0/"

def main():
    print(load_directory)

    T = 300 # [K]
    root_save_dir = "./data/thermal/"

    # load files
    config = toml.load(load_directory+"config.toml")
    with open(load_directory+'/j_best.txt', 'r') as file:
        content = file.read()
    j_read = [float(val) for val in re.findall(r'-?\d+\.\d+e[+-]?\d+', content)]

    # シミュレーション設定
    end = 2.0e-9 # シミュレーションの終了時間 [秒]
    dt = config["simulation"]["dt"]*1e-9 # タイムステップ [秒]
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
    H_shape =np.array(config["fields"]["H_shape"]) # 反磁場 [T]
    j = np.zeros(steps, dtype=np.float64)
    j[:len(j_read)] = j_read

    m0 = np.array(config["simulation"]["m0"])  # 初期磁化
    current = config["simulation"]["current"] # 印加電流密度

    save_dir = root_save_dir+f"M{M*1e-3:.0f}/J{current:04.1f}e10_T{T:.0f}/"   # 結果を保存するディレクトリ名
    os.makedirs(save_dir, exist_ok=True)   # 結果を保存するディレクトリ

    system = s.ThermalSystem(end, dt, alphaG, beta, theta, size, d_Pt, M, H_appl, H_ani, m0, T)
    system.j = j
    system.output(save_dir+"config.toml")

    # シミュレーション実行
    for n in range(1, 1001):
        print(n)
        system.reset()

        system.run()

        # 結果を保存
        label = f"{n:03d}"
        system.save_episode(label, save_dir)
        np.savetxt(save_dir+label+".txt", system.m)

if __name__ == '__main__':
    main()
