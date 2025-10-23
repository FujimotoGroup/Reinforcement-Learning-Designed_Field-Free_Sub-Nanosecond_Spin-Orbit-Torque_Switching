import math
import numpy as np
import toml
import matplotlib as mpl
#mpl.use('Agg')  # 非対話型バックエンドを指定
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

import mpmath as mp

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
                "demag": self.demag.tolist() if isinstance(self.demag, np.ndarray) else np.float64(self.demag),
            },
            "geometry": {
                "size": self.size.tolist(),
                "V": float(self.V),
            },
            "fields": {
                "H_appl": self.H_appl.tolist(),
                "H_ani": self.H_ani.tolist(),
                "H_shape": self.H_shape.tolist() if isinstance(self.H_shape, np.ndarray) else np.float64(self.H_shape),
                "H_s": float(self.H_s),
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

    def energy_i(self, i):
        H_eff = self.H_appl + self.H_ani * self.m[i] - self.H_shape * self.m[i]
        H_eff += self.H_s * self.j[i] * np.cross(self.e_y, self.m[i]) + self.beta * self.H_s * self.j[i] * self.e_y
        H_eff += self.H_rf * self.j[i]
        energy = - self.mu0*self.M*np.dot(self.m[i], H_eff)
        energy -= self.mu0*self.M*self.H_shape[1]
        return energy

    def energy(self):
        unit = np.ones(self.m.shape)
        jj = self.j[:, None]
        H_eff = self.H_appl*unit + self.H_ani * self.m - self.H_shape * self.m
        H_eff += self.H_s * jj * np.cross(self.e_y*unit, self.m) + self.beta * self.H_s * jj * self.e_y
        H_eff += self.H_rf * jj
        energy = - self.mu0*self.M*np.sum(self.m*H_eff, axis=1)
        energy -= self.mu0*self.M*self.H_shape[1]
        return energy

    def eta_1(self, current, t):
        a11 = - self.alphaG * (self.H_shape[1] - self.H_shape[0])
        a12 =    self.H_shape[2] - self.H_shape[0]
        a21 = - (self.H_shape[1] - self.H_shape[0])
        a22 = - self.alphaG * (self.H_shape[2] - self.H_shape[0])
        A = np.array([[a11, a12], [a21, a22]])
        values, vectors = np.linalg.eig(A)
#        print(values*self.g)
        P = vectors
        P_inv = np.linalg.inv(P)

        e_x = np.array([1e0, 0e0, 0e0])
        h0 = np.array([0e0, self.H_rf[1]+self.beta*self.H_s, - self.H_s])
        h0 = h0 * current
        exh0 = np.cross(e_x, h0)
        exexh0 = np.cross(e_x, exh0)
        f = exh0 + self.alphaG*exexh0
        f_1 = f[1:3]

        D = np.zeros((len(t), 2, 2), dtype=np.complex128)
        D[:, 0, 0] = np.exp(-self.g*values[0]*t)/values[0] - 1e0/values[0]
        D[:, 1, 1] = np.exp(-self.g*values[1]*t)/values[1] - 1e0/values[1]

        eta_1 = np.dot(P@D@P_inv, f_1)

        n_terms = 2 # テイラー展開次数

        coeffs = []
        Af = f_1.copy()
        sign = -1
        for n in range(1, n_terms+1):
            term = sign * Af / math.factorial(n)
            coeffs.append(term)
            Af = np.dot(A, Af)  # Aの累乗を計算
            sign *= -1

        # 多項式評価用の配列作成
        # coeffs[i]は t**(i+1) の係数
        # np.polynomial.polynomial.polyvalは定数項が最初なので0を追加
        coeffs_array = np.stack(coeffs, axis=0)  # (n_terms, 2)

        # eta_1 = sum_{n=1}^n_terms coeffs[n-1] * t**n
        # ⇒定数項0の多項式としてpolyvalを使う
        t_powers = np.vstack([(self.g*t)**i for i in range(1, n_terms+1)])  # (n_terms, len(t))
        eta_1 = np.dot(coeffs_array.T, t_powers).T

        N = - np.array([[0,0,0],[0,self.H_shape[1]-self.H_shape[0],0],[0,0,self.H_shape[2]-self.H_shape[0]]])

        f1xNf1 = np.cross(f, np.dot(N,f))
        Hsf1xyxf1 = self.H_s * np.cross(f, np.cross(self.e_y, f))

        A_1 = np.array([[0,0,0],[0,a11,a12],[0,a21,a22]])
        Af1xNf1 = np.cross(np.dot(A_1,f), np.dot(N,f))
        HsAf1xyxf1 = self.H_s * np.cross(np.dot(A_1,f), np.cross(self.e_y, f))
        f1xNAf1 = np.cross(f, np.dot(N,np.dot(A_1,f)))
        Hsf1xyxAf1 = self.H_s * np.cross(f, np.cross(self.e_y, np.dot(A_1,f)))

        Af1xNAf1 = np.cross(np.dot(A_1,f), np.dot(N,np.dot(A_1,f)))
        HsAf1xyxAf1 = self.H_s * np.cross(np.dot(A_1,f), np.cross(self.e_y, np.dot(A_1,f)))

        f1sq = np.dot(f_1, f_1)
        A1f1 = np.dot(A, f_1)
        f1A1f1 = np.dot(f_1, A1f1)
        A1f1sq = np.dot(A1f1, A1f1)

        d2 = -f1sq/2e0*exh0 + f1xNf1 + Hsf1xyxf1
        d3 = -f1A1f1*exh0 + Af1xNf1 + f1xNAf1 + HsAf1xyxf1 + Hsf1xyxAf1
        d4 = -A1f1sq/2e0*exh0 + Af1xNAf1 + HsAf1xyxAf1
        d2 += -f1sq*exexh0 + self.alphaG * np.cross(e_x, d2)
        d3 += -2e0*f1A1f1*exexh0 + self.alphaG * np.cross(e_x, d3)
        d4 += -A1f1sq*exexh0 + self.alphaG * np.cross(e_x, d4)
        d2 = - d2[1:3] / 3e0
        d3 =   d3[1:3] / 8e0
        d4 = - d4[1:3] /20e0

#        print(f_1,d2,d3,d4)
#        print(self.g*t[:40])

        xi_1 = np.zeros((len(t), 2))
        xi_1[:,0] = d2[0]*(self.g*t)**3 + d3[0]*(self.g*t)**4 + d4[0]*(self.g*t)**5
        xi_1[:,1] = d2[1]*(self.g*t)**3 + d3[1]*(self.g*t)**4 + d4[1]*(self.g*t)**5
#        print(xi_1[:40])
        eta_1 = eta_1# + xi_1

        return eta_1


    def first(self, current):
        eta = self.eta_1(current, self.t)
        return eta

    def plot(self):
        t = np.concatenate(([-self.dt], self.t))
        m = np.concatenate(([self.m0], self.m))
        j = np.concatenate(([0e0], self.j))
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].set_ylim([-1e0,1e0])
        axes[0].plot(t, m[:,0], label=r"$m_x$")
        axes[0].plot(t, m[:,1], label=r"$m_y$")
        axes[0].plot(t, m[:,2], label=r"$m_z$")
        axes[0].set_xlabel('Time (ns)')
        axes[0].set_ylabel('Magnetization')
        axes[0].legend()

        axes[1].plot(t, j, color='gold')
        axes[1].set_xlabel('Time (ns)')
        axes[1].set_ylabel(r"Current Density (MA/cm$^2$)")

        return fig

    def plot_energy(self):
        t = np.concatenate(([-self.dt], self.t))
        m = np.concatenate(([self.m0], self.m))
        j = np.concatenate(([0e0], self.j))
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].set_ylim([-1e0,1e0])
        axes[0].plot(t, m[:,0], label=r"$m_x$")
        axes[0].plot(t, m[:,1], label=r"$m_y$")
        axes[0].plot(t, m[:,2], label=r"$m_z$")
        axes[0].set_xlabel('Time (ns)')
        axes[0].set_ylabel('Magnetization')
        axes[0].legend()

        # 1の区間を検出してハイライト
        in_block = False
        start = 0
        jj = np.concatenate((j, [0e0]))
        mask = jj > 0e0
        for i, val in enumerate(mask):  # 番兵として末尾に0を追加
            if val == 1 and not in_block:
                start = i
                in_block = True
            elif val == 0 and in_block:
                axes[0].axvspan(start*self.dt, i*self.dt, color='yellow', alpha=0.3)
                axes[1].axvspan(start*self.dt, i*self.dt, color='yellow', alpha=0.3)
                in_block = False

        axes[1].plot(self.t, self.energy(), color='black')
        axes[1].set_xlabel('Time (ns)')
        axes[1].set_ylabel("Energy")

        axes[1].axhline(y=0, color="black", ls=":")

        return fig, axes

    def save_data(self, label:str, directory:str):
        np.savetxt(directory+"m_"+label+".txt", self.m)
        np.savetxt(directory+"j_"+label+".txt", self.j)

    def save_episode(self, label:str, directory:str):
        fig = self.plot()
        fig.savefig(directory+label+".png", dpi=200)

        fig, _ = self.plot_energy()
        fig.savefig(directory+label+"_energy.png", dpi=200)
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
    T = 300 # 温度 [K]
    end = 2.0e-9  # シミュレーションの終了時間 [秒]
    dt = 1e-12  # タイムステップ [秒]
    alphaG = 0.015e0  # ギルバート減衰定数
    beta = -3e0  # field like torque と damping like torque の比
    theta = -0.25e0  # スピンホール角
    size = np.array([100e-9, 50e-9, 1e-9]) # [m] 強磁性体の寸法
#    size = np.array([80e-9, 20e-9, 1e-9]) # [m] 強磁性体の寸法
#    j0 = 2.64789e0 # alphaG = 0.01
#    j0 = 3e0 # alphaG = 0.01
#    j0 = 2.405e0 # alphaG = 0
    j0 = 10e0
#    size = np.array([80e-9, 25e-9, 1e-9]) # [m] 強磁性体の寸法
#    j0 = 6.35e0
#    j0 = 7e0
    d_Pt = 5.0e-9  # Ptの厚み [m]
#    M = 750e3  # 飽和磁化　[A/m]
    M = 1500e3  # 飽和磁化　[A/m]

    H_appl = np.array([0e0, 0e0, 0e0])  # 外部磁場 [T]
    H_ani = np.array([0e0, 0e0, 0e0])  # 異方性定数 [T]
    m0 = np.array([1e0, 0e0, 0e0])  # 初期磁化
    system = ThermalSystem(end, dt, alphaG, beta, theta, size, d_Pt, M, H_appl, H_ani, m0, T)

    coef = system.H_rf[1] + system.beta*system.H_s
    j_th = (system.H_shape[1] - system.H_shape[0]) / coef
    print(j_th, j0)

    i_t1_off = -1
    j = j0 # [MA/cm2]
    triger1 = False
    triger2 = False

    A = system.g * system.beta * system.H_s * j0
    B = system.g*(system.H_shape[2]-system.H_shape[0])
    C = (system.H_shape[2]-system.H_shape[0])/(system.H_shape[1]-system.H_shape[0])
    omega2 = system.g*np.sqrt((system.H_shape[2]-system.H_shape[1])*(system.H_shape[2]-system.H_shape[0]))
    print("1/omega2 = ", 1e0/omega2)
    deltaE = 2e0*system.alphaG*np.sqrt( (system.demag[2]-system.demag[1])*(system.demag[1]-system.demag[0]) )
    print("delta e = ", deltaE)
    t0_approx =  np.sqrt(2e0)/omega2*np.sqrt(-1e0 + np.sqrt(1e0 + omega2**2/C/A**2*(1e0+deltaE/(system.H_shape[1]-system.H_shape[0]))))

    coeffs = np.array([1e0, -4e0/B, 4e0/B**2*(1e0+C), 0e0, -4e0/(A*B)**2])
    roots = np.roots(coeffs)
    t0 = 0e0
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > 0:
            t0 = r.real
    i_t0 = np.where(system.t > t0)[0][0]

    eta_1 = system.eta_1(j0, np.array([t0]))[0]
    eta_1 = np.insert(eta_1, 0, np.sqrt(1e0-np.dot(eta_1,eta_1)))
    e2 = np.dot(eta_1, system.H_shape*eta_1)/system.mu0M + deltaE
    Nx, Ny, Nz = system.demag
    print(np.dot(eta_1, system.H_shape*eta_1)/system.mu0M, e2, Ny)
    u = np.sqrt((e2 - Nx)/(Nz - Nx))
    l = np.sqrt((e2 - Ny)/(Nz - Ny))
    k = 1e0 - (l/u)**2
    c = 2e0*mp.ellipk(k)/(u*omega2)

    for i in range(system.steps):
        if not(triger1) and system.energy_i(i) > 0e0+deltaE:
#        if system.energy_i(i) > 0e0+0.002:
#        if not(triger1) and system.t[i] > t0+5*system.dt:
            triger1 = True
            j = 0
            print("approx t0: ", t0_approx)
            print("exact  t0: ", t0, "diff: ", t0-t0_approx, "time step diff: ", (t0-t0_approx)/system.dt)
            print("actual t0: ", system.t[i], (t0-system.t[i])/system.dt)
            i_t1_off = i
        if not(triger2) and system.m[system.i,0] < -0.85:
            j = j0
            triger2 = True
            print("approx t2: ", c+t0, c)
            print("actual t2: ", system.t[system.i])
        if triger2 and np.abs(system.m[system.i,2]) < 0.01:
            j = 0
            tirger2 = False
        system.RungeKutta(j)  # 磁化の時間発展を計算

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    epsilon2 = system.demag[1] + (2e0*system.demag[1]-system.demag[1])*np.linspace(0,1,1000)
    E = []
    for e in epsilon2:
        Nx, Ny, Nz = system.demag
        u = np.sqrt((e - Nx)/(Nz - Nx))
        l = np.sqrt((e - Ny)/(Nz - Ny))
        k = 1e0 - (l/u)**2
        c = 2e0*mp.ellipk(k)/(u*omega2)
        E.append(c)
    E = np.array(E)
    ax.plot(epsilon2,E)
    ax.axvline(x=e2,c="black")
    e2_0 = system.energy_i(i_t1_off)/system.mu0M + Ny
    u = np.sqrt((e2_0 - Nx)/(Nz - Nx))
    l = np.sqrt((e2_0 - Ny)/(Nz - Ny))
    k = 1e0 - (l/u)**2
    c = 2e0*mp.ellipk(k)/(u*omega2)
    ax.axvline(x=e2_0,c="blue")
    ax.axhline(y=c,c="red")
    plt.show()


    fig, axes = system.plot_energy()
    print(i_t0, system.energy_i(i_t0+3))

    j = j0 # [MA/cm2]
    system.reset()
    for i in range(system.steps):
        system.RungeKutta(j)

    eta_1 = system.first(j0)
    n = i_t1_off
    axes[0].plot(system.t[:n], np.real(eta_1[:n,0]), color="purple")
    axes[0].plot(system.t[:n], np.real(eta_1[:n,1]), color="blue")

    axes[1].plot(system.t, system.energy())
    fig.tight_layout()
    plt.show()
    fig.savefig("test.png", dpi=200)

    i_z = np.argmin(eta_1[:i_t1_off,1])
    print(i_z)

    j = j0 # [MA/cm2]
    system2 = ThermalSystem(end, dt, alphaG, beta, theta, size, d_Pt, M, H_appl, H_ani, m0, T)
    triger = False
    for i in range(system2.steps):
        if i > i_z:
            j = 0
        if not(triger) and system2.m[system2.i,0] < -0.85:
            j = j0
            triger = True
        if triger and np.abs(system2.m[system2.i,2]) < 0.01:
            j = 0
            tirger = False
        system2.RungeKutta(j)

    fig, axes = system2.plot_energy()

    axes[1].plot(system.t, system.energy())
    fig.tight_layout()
    fig.savefig("test2.png", dpi=200)

#    label = "test2"
#    system2.save_episode(label, "./")
    system.output()
