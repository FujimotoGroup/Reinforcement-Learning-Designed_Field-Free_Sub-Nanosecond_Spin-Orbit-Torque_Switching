import numpy as np
import toml

import matplotlib as mpl
mpl.use('Agg')  # 非対話型バックエンドを指定
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load(directory):
    print(directory)

    # load files
    config = toml.load(directory+"config.toml")
    t = np.loadtxt(directory+'t.txt')
    m = np.loadtxt(directory+'m_best.txt')
    j = np.loadtxt(directory+'j_best.txt')

    dt = config["simulation"]["dt"]
    m0 = np.array(config["simulation"]["m0"])

    t = np.concatenate(([-dt], t))
    m = np.concatenate(([m0], m))
    j = np.concatenate(([0e0], j))

    return config, t, m, j

def plot_sphere(ax):
    # --- 単位球のメッシュ生成 ---
    u = np.linspace(0, 2*np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    # x = cos(u) sin(v), y = sin(u) sin(v), z = cos(v)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    # 1) 単位球を半透明で描画
    ax.plot_surface(
        xs, ys, zs,
        rstride=4, cstride=4,        # メッシュの粗さ
        color='lightgray',           # 球の色
        alpha=0.2,                   # 透明度（0.0～1.0）
        edgecolor='none'
    )

def plot_line_with_alpha_segments(ax, x, y, z, alpha_mask, front_alpha=0.3, back_alpha=0.05, color='gray', linewidth=0.5):
    """
    alpha_mask: Trueならfront, Falseならback
    front_alpha / back_alpha: 透過度
    """
    # 前面・背面それぞれの線をNaN挟みで分割
    def segment_masked_line(mask, x, y, z):
        x_seg = np.where(mask, x, np.nan)
        y_seg = np.where(mask, y, np.nan)
        z_seg = np.where(mask, z, np.nan)
        return x_seg, y_seg, z_seg

    # 背面
    x_back, y_back, z_back = segment_masked_line(~alpha_mask, x, y, z)
    ax.plot(x_back, y_back, z_back, color=color, alpha=back_alpha, linewidth=linewidth)

    # 前面
    x_front, y_front, z_front = segment_masked_line(alpha_mask, x, y, z)
    ax.plot(x_front, y_front, z_front, color=color, alpha=front_alpha, linewidth=linewidth)


def plot_sphere_with_depth_wire(ax, elev=30, azim=45, wire_u_step=10, wire_v_step=10):
    """
    表面を滑らかに、ワイヤフレームは間引いて前面/背面描画
    """
    # 高解像度メッシュ（表面用）
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    u_grid, v_grid = np.meshgrid(u, v)

    xs = np.cos(u_grid) * np.sin(v_grid)
    ys = np.sin(u_grid) * np.sin(v_grid)
    zs = np.cos(v_grid)

    # 視点ベクトル
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    view_vec = np.array([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad)
    ])

    # 内積で奥行き判定
    dot = xs * view_vec[0] + ys * view_vec[1] + zs * view_vec[2]
    front_mask = dot >= 0
    xs_masked = np.where(front_mask, xs, np.nan)
    ys_masked = np.where(front_mask, ys, np.nan)
    zs_masked = np.where(front_mask, zs, np.nan)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()
    axis(ax, [x_min, 0, 0], [x_max, 0, 0], color='gray', alpha=0.3)
    axis(ax, [0, y_min, 0], [0, y_max, 0], color='gray', alpha=0.3)
    axis(ax, [0, 0, z_min], [0, 0, z_max], color='gray', alpha=0.3)

    # 前面だけ塗りつぶし
    ax.plot_surface(xs_masked, ys_masked, zs_masked, color='lightgray', alpha=0.2, edgecolor='none')

    # --- 間引いてワイヤフレームを描画 ---
    # 経線（縦線）u方向
    for i in range(0, xs.shape[0], wire_u_step):
        x_line = xs[i, :]
        y_line = ys[i, :]
        z_line = zs[i, :]
        is_front = dot[i, :] >= 0
        plot_line_with_alpha_segments(ax, x_line, y_line, z_line, is_front)

    # 緯線（横線）v方向
    for j in range(0, xs.shape[1], wire_v_step):
        x_line = xs[:, j]
        y_line = ys[:, j]
        z_line = zs[:, j]
        is_front = dot[:, j] >= 0
        plot_line_with_alpha_segments(ax, x_line, y_line, z_line, is_front)

    # 視点
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])

def arrow(ax, direction, length=1.0, color='red', alpha=1.0, shaft_radius=0.02, head_radius=0.05, head_length=0.15):
    """
    原点から始まる3D矢印を描画する（Mathematica風）

    Parameters:
        ax: matplotlib の 3D Axes
        direction: 矢印の向き (x, y, z) のタプルまたは配列
        length: 矢印の長さ（単位ベクトルをスケーリング）
        color: 矢印の色
        alpha: 透明度（0:完全に透明, 1:不透明）
        shaft_radius: 矢印の棒の半径
        head_radius: 矢印の先端部分の半径
        head_length: 矢印の先端部分の長さ
    """
    direction = np.array(direction)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("方向ベクトルはゼロであってはいけません")
    direction = direction / norm * length  # 単位ベクトルにしてからスケーリング

    # 原点から先端までの座標
    start = np.array([0, 0, 0])
    end = direction

    # 棒の長さ = 全長 - 矢印頭長さ
    shaft_length = length - head_length
    shaft_end = direction * (shaft_length / length)

    # 円柱（棒）を描く
    def plot_cylinder(start, end, radius, color, alpha):
        v = end - start
        mag = np.linalg.norm(v)
        v = v / mag
        not_v = np.array([1, 0, 0]) if not np.allclose(v, [1, 0, 0]) else np.array([0, 1, 0])
        n1 = np.cross(v, not_v)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(v, n1)
        t = np.linspace(0, 2 * np.pi, 20)
        circle = np.array([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)])
        cylinder = []
        for i in [0, mag]:
            pt = start + v * i
            circle_pts = pt[:, None] + n1[:, None] * circle[0] + n2[:, None] * circle[1]
            cylinder.append(circle_pts.T)
        for i in range(len(t) - 1):
            verts = [cylinder[0][i], cylinder[0][i+1], cylinder[1][i+1], cylinder[1][i]]
            ax.add_collection3d(Poly3DCollection([verts], color=color, alpha=alpha))

    # 円錐（矢印の頭）を描く
    def plot_cone(start, direction, radius, length, color, alpha):
        v = direction / np.linalg.norm(direction)
        not_v = np.array([1, 0, 0]) if not np.allclose(v, [1, 0, 0]) else np.array([0, 1, 0])
        n1 = np.cross(v, not_v)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(v, n1)
        t = np.linspace(0, 2 * np.pi, 20)
        circle = np.array([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)])
        base_center = start
        tip = start + v * length
        circle_pts = base_center[:, None] + n1[:, None] * circle[0] + n2[:, None] * circle[1]
        for i in range(len(t) - 1):
            verts = [circle_pts[:, i], circle_pts[:, i+1], tip]
            ax.add_collection3d(Poly3DCollection([verts], color=color, alpha=alpha))

    # 棒を描画
    plot_cylinder(start, shaft_end, shaft_radius, color, alpha)

    # 矢印の頭を描画
    plot_cone(shaft_end, end - shaft_end, head_radius, head_length, color, alpha)

def axis(ax, start, end, color='black', alpha=1.0, shaft_radius=0.01, head_radius=0.03, head_length=0.1):
    """
    任意の始点と終点で3D矢印を描く（軸表示などに使う）
    """
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    if length == 0:
        return
    direction = direction / length  # 単位ベクトル
    shaft_length = length - head_length
    shaft_end = start + direction * shaft_length

    # 内部関数：円柱と円錐を描画（前の回答の内容と同じ）
    def plot_cylinder(start, end, radius, color, alpha):
        v = end - start
        mag = np.linalg.norm(v)
        v = v / mag
        not_v = np.array([1, 0, 0]) if not np.allclose(v, [1, 0, 0]) else np.array([0, 1, 0])
        n1 = np.cross(v, not_v)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(v, n1)
        t = np.linspace(0, 2 * np.pi, 20)
        circle = np.array([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)])
        cylinder = []
        for i in [0, mag]:
            pt = start + v * i
            circle_pts = pt[:, None] + n1[:, None] * circle[0] + n2[:, None] * circle[1]
            cylinder.append(circle_pts.T)
        for i in range(len(t) - 1):
            verts = [cylinder[0][i], cylinder[0][i+1], cylinder[1][i+1], cylinder[1][i]]
            ax.add_collection3d(Poly3DCollection([verts], color=color, alpha=alpha))

    def plot_cone(start, direction, radius, length, color, alpha):
        v = direction / np.linalg.norm(direction)
        not_v = np.array([1, 0, 0]) if not np.allclose(v, [1, 0, 0]) else np.array([0, 1, 0])
        n1 = np.cross(v, not_v)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(v, n1)
        t = np.linspace(0, 2 * np.pi, 20)
        circle = np.array([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)])
        base_center = start
        tip = start + v * length
        circle_pts = base_center[:, None] + n1[:, None] * circle[0] + n2[:, None] * circle[1]
        for i in range(len(t) - 1):
            verts = [circle_pts[:, i], circle_pts[:, i+1], tip]
            ax.add_collection3d(Poly3DCollection([verts], color=color, alpha=alpha))

    plot_cylinder(start, shaft_end, shaft_radius, color, alpha)
    plot_cone(shaft_end, end - shaft_end, head_radius, head_length, color, alpha)
def main():
    load_dir = "../data/100x50x1/aG0.010/M750/J03.0e10_T0/"
    config, t, m, j = load(load_dir)
    print(config)

if __name__ == '__main__':
    main()
