import numpy as np
import toml
import matplotlib as mpl
mpl.use('Agg')  # 非対話型バックエンドを指定
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({
    "text.usetex": False,
})
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams["font.size"] = 20

class System:
    def __init__(self, end:np.float64, dt:np.float64, alphaG:np.float64, beta:np.float64, theta:np.float64, size:np.array, d_Pt, M:np.float64,
                 H_appl:np.array, H_ani:np.array, m0:np.array):

        self.gamma = 1.760859770e11 # [rad/s T]
        self.planck = 1.054571817e-34  # [J s] = [C V s] = [C T m2]
        self.mu0 = 4e0*np.pi*1e-7 # [H/m] = [T/(A/m)]
        self.e = 1.602176634e-19  # [C]
        self.e_y = np.array([0e0, 1e0, 0e0])
        self.alphaG = alphaG # [-]
        self.beta = beta # [-]
        self.theta = theta # [-]
        self.m0 = m0 # [-]
        self.size = size # [[m], [m], [m]]
        self.V = size[0]*size[1]*size[2] # [m3]
        self.d_Pt = d_Pt # [nm]
        self.M = M # [A/m]
        self.H_appl = H_appl # [T]
        self.H_ani = H_ani # [T]

        self.demag = self.Demag(*size) # [-]
        self.mu0M = self.mu0 * self.M # [T]
        self.H_shape = self.mu0M * self.demag # [T]
        self.H_s = (self.planck * self.theta) / (2e0 * self.e * self.M * self.size[2]) *1e10 # [T / (MA/cm2)]
        self.H_rf = np.array([0e0, - self.mu0 * self.d_Pt / 2e0, 0e0]) *1e10  # [T / (MA/cm2)]
        self.g = self.gamma*1e-9 / (1e0 + self.alphaG**2)

        self.end = end*1e9  # simulation end time [ns]
        self.dt = dt*1e9 # [ns]
        self.steps = int(self.end / self.dt)  # total time step
        self.t = np.linspace(0e0, self.end, self.steps)
        self.m = np.zeros((self.steps, 3), dtype=np.float64)
        self.j = np.zeros(self.steps, dtype=np.float64)
        self.steps = self.steps - 1 # 調整

        self.i = 0 # time counter
        self.m[0,:] = m0

    def run(self):
        for i in range(self.steps):
            self.RungeKutta(self.j[i])

    def judge(self, current:np.array, rule_period:np.float64, rule_range:np.float64):
        self.j[:] = 0e0
        l = min(self.steps, len(current))
        self.j[:l] = current[:l]

        self.run()

        period = np.where((self.t > rule_period*1e9) & (self.t < self.t[-1]))[0]
        condition = (self.m[period,0] < rule_range).all()

#        violating_indices = period[self.m[period, 0] >= rule_range]
#        print("違反時刻:", self.t[violating_indices])
#        print("x成分:", self.m[violating_indices, 0])

        return condition

    def set(self, end):
        self.end = end*1e9  # simulation end time [ns]
        self.steps = int(self.end / self.dt)  # total time step
        self.t = np.linspace(0e0, self.end, self.steps)
        self.m = np.empty((self.steps, 3), dtype=np.float64)
        self.j = np.empty(self.steps, dtype=np.float64)
        self.steps = self.steps - 1 # 調整
        self.reset()

    def reset(self):
        self.i = 0
        self.m[0,:] = self.m0

    def getConfig(self):
        data = {
            "simulation": {
                "end": self.end,
                "dt": self.dt,
                "steps": self.steps,
                "m0": self.m0.tolist(),
            },
            "material": {
                "alphaG": self.alphaG,
                "beta": self.beta,
                "theta": self.theta,
                "M": self.M,
                "d_Pt": self.d_Pt,
                "e_y": self.e_y.tolist(),
            },
            "geometry": {
                "size": self.size.tolist(),
                "V": float(self.V),
            },
            "fields": {
                "H_appl": self.H_appl.tolist(),
                "H_ani": self.H_ani.tolist(),
                "H_shape": self.H_shape.tolist() if isinstance(self.H_shape, np.ndarray) else np.float64(self.H_shape),
                "H_s": self.H_s,
                "H_rf": self.H_rf.tolist(),
            },
            "constants": {
                "gamma": self.gamma,
                "planck": self.planck,
                "mu0": self.mu0,
                "e": self.e,
                "g": self.g,
            }
        }

        return data

    def output(self, file:str = "config.toml"):
        data = self.getConfig()
        with open(file, 'w') as f:
            toml.dump(data, f)

    # 減磁率 Dx, Dy, Dz を計算
    def Demag(self, a, b, c):
        def D(a, b, c):
            term1 = (b**2 - c**2) / (2 * b * c) * np.log(abs((np.sqrt(a**2 + b**2 + c**2) - a) / (np.sqrt(a**2 + b**2 + c**2) + a)))
            term2 = (a**2 - c**2) / (2 * a * c) * np.log(abs((np.sqrt(a**2 + b**2 + c**2) - b) / (np.sqrt(a**2 + b**2 + c**2) + b)))
            term3 = (b / (2 * c)) * np.log(abs((np.sqrt(a**2 + b**2) + a) / (np.sqrt(a**2 + b**2) - a)))
            term4 = (a / (2 * c)) * np.log(abs((np.sqrt(a**2 + b**2) + b) / (np.sqrt(a**2 + b**2) - b)))
            term5 = (c / (2 * a)) * np.log(abs((np.sqrt(b**2 + c**2) - b) / (np.sqrt(b**2 + c**2) + b)))
            term6 = (c / (2 * b)) * np.log(abs((np.sqrt(a**2 + c**2) - a) / (np.sqrt(a**2 + c**2) + a)))
            term7 = 2 * np.arctan((a * b) / (c * np.sqrt(a**2 + b**2 + c**2)))  # 符号変更
            term8 = (a**3 + b**3 - 2 * c**3) / (3 * a * b * c)
            term9 = (a**2 + b**2 - 2 * c**2) / (3 * a * b * c) * np.sqrt(a**2 + b**2 + c**2)
            term10 = c / (a * b) * (np.sqrt(a**2 + c**2) + np.sqrt(b**2 + c**2))
            term11 = - ((a**2 + b**2)**(3/2) + (b**2 + c**2)**(3/2) + (c**2 + a**2)**(3/2)) / (3 * a * b * c)

            return (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11) / np.pi

        # 各方向の減磁率を計算
        Dz = D(a, b, c)  # 既存の計算
        Dx = D(b, c, a)  # a→b, b→c, c→a
        Dy = D(c, a, b)  # a→c, b→a, c→b

        return np.array([Dx, Dy, Dz])  # 3次元ベクトルとして返す

    def LLG(self, magnetization:np.array, current:np.float64) -> np.array:
        H_eff = self.H_appl + self.H_ani * magnetization - self.H_shape * magnetization
        H_eff += self.H_s * current * np.cross(self.e_y, magnetization) + self.beta * self.H_s * current * self.e_y
        H_eff += self.H_rf * current
        mxH = np.cross(magnetization, H_eff)
        m = - self.g * mxH - self.g * self.alphaG * np.cross(magnetization, mxH)
        return m

    def RungeKutta(self, current:np.float64):
        self.j[self.i] = current
        m1 = self.LLG(self.m[self.i],                  current)
        m2 = self.LLG(self.m[self.i] + self.dt*m1/2e0, current)
        m3 = self.LLG(self.m[self.i] + self.dt*m2/2e0, current)
        m4 = self.LLG(self.m[self.i] + self.dt*m3,     current)
        self.m[self.i+1] = self.m[self.i] + self.dt/6e0 * (m1 + 2e0*m2 + 2e0*m3 + m4)
        self.i += 1

    def plot(self):
        t = np.concatenate(([-self.dt], self.t))
        m = np.concatenate(([self.m0], self.m))
        j = np.concatenate(([0e0], self.j))
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].set_ylim([-1e0,1e0])
        axes[0].plot(t, m[:,0], label='$m_x$')
        axes[0].plot(t, m[:,1], label='$m_y$')
        axes[0].plot(t, m[:,2], label='$m_z$')
        axes[0].set_xlabel('Time (ns)')
        axes[0].set_ylabel('Magnetization')
        axes[0].legend()

        axes[1].plot(t, j, color='gold')
        axes[1].set_xlabel('Time (ns)')
        axes[1].set_ylabel('Current Density (MA/cm$^2$)')

        return fig

    def save_data(self, label:str, directory:str):
        np.savetxt(directory+"m_"+label+".txt", self.m)
        np.savetxt(directory+"j_"+label+".txt", self.j)

    def save_episode(self, label:str, directory:str):
        fig = self.plot()
        fig.tight_layout()
        fig.savefig(directory+label+".png", dpi=200)
        plt.close(fig)

class ThermalSystem(System):
    def __init__(self, end:np.float64, dt:np.float64, alphaG:np.float64, beta:np.float64, theta:np.float64, size:np.array, d_Pt:np.float64, M:np.float64,
                 H_appl:np.array, H_ani:np.array, m0:np.array, T:np.float64):
        super().__init__(end, dt, alphaG, beta, theta, size, d_Pt, M, H_appl, H_ani, m0)

        self.kB = 1.380649e-23 # [J/K]
        self.T = T # [K]
        self.H_th = np.sqrt((2e0 * self.alphaG * self.kB * self.T) / (self.M * self.gamma * self.V * (self.dt*1e-9))) # [T]

    def LLB(self, magnetization:np.array, current:np.float64, H_therm) -> np.array:
        H_eff = self.H_appl + self.H_ani * magnetization - self.H_shape * magnetization
        H_eff += self.H_s * current * np.cross(self.e_y, magnetization) + self.beta * self.H_s * current * self.e_y
        H_eff += self.H_rf * current
        H_eff += H_therm
        mxH = np.cross(magnetization, H_eff)
        m = - self.g * mxH - self.g * self.alphaG * np.cross(magnetization, mxH)
        return m

    def RungeKutta(self, current:np.float64):
        self.j[self.i] = current
        H_therm = self.H_th * np.random.normal(0e0, 1e0, 3)
        m1 = self.LLB(self.m[self.i],                  current, H_therm)
        m2 = self.LLB(self.m[self.i] + self.dt*m1/2e0, current, H_therm)
        m3 = self.LLB(self.m[self.i] + self.dt*m2/2e0, current, H_therm)
        m4 = self.LLB(self.m[self.i] + self.dt*m3,     current, H_therm)
        self.m[self.i+1] = self.m[self.i] + self.dt/6e0 * (m1 + 2e0*m2 + 2e0*m3 + m4)
        self.i += 1

    def run(self):
        for i in range(self.steps):
            self.RungeKutta(self.j[i])

    def getStability(self, T):
        E_ani = (self.demag[0] - self.demag[1]) * self.M**2 * self.mu0 / 2e0
        stability = np.nan
        if T != 0:
            stability = (E_ani * self.V)/(self.kB * T)
        return stability

    def getConfig(self):
        data = super().getConfig()
        data["constants"]["kB"] = self.kB
        data["simulation"]["T"] = self.T
        data["fields"]["h_th"] = float(self.H_th)
        data["material"]["thermal_stability"] = self.getStability(self.T)
        return data

if __name__ == '__main__':
    # シミュレーション設定
    T = 0 # 温度 [K]
    end = 0.8e-9  # シミュレーションの終了時間 [秒]
    dt = 1e-12  # タイムステップ [秒]
    alphaG = 0.01e0  # ギルバート減衰定数
    beta = -3e0  # field like torque と damping like torque の比
    theta = -0.25e0  # スピンホール角
    size = np.array([100e-9, 50e-9, 1e-9]) # [m] 強磁性体の寸法
    d_Pt = 5.0e-9  # Ptの厚み [m]
    M = 750e3  # 飽和磁化　[A/m]
    H_appl = np.array([0e0, 0e0, 0e0])  # 外部磁場 [T]
    H_ani = np.array([0e0, 0e0, 0e0])  # 異方性定数 [T]
    m0 = np.array([1e0, 0e0, 0e0])  # 初期磁化
    system = ThermalSystem(end, dt, alphaG, beta, theta, size, d_Pt, M, H_appl, H_ani, m0, T)

    for i in range(system.steps):
        j = 10e0 # [MA/cm2]
        system.RungeKutta(j)  # 磁化の時間発展を計算

    label = "test"
    system.save_episode(label, "./")
    system.output()
