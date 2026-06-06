import numpy as np
import matplotlib.pyplot as plt
# 20260524 shimojo & Gemini
# ==========================================
# 1. 形状・物性値・計算条件の設定
# ==========================================
L = 0.09          # 壁の厚み [m] (9cm)
A = 1.0          # 断面積 [m²] (1m x 1m)
T_init = 20.0     # 壁の初期温度 [°C]

# 左右それぞれの周囲環境温度
T_env_left = 50.0   # 左側の周囲温度 [°C] (温風などを想定)
T_env_right = 0.0   # 右側の周囲温度 [°C] (冷風・室温などを想定)

# 木材の物性値（壁の木材；針葉樹）
k = 0.12          # 熱伝導率 [W/(m·K)]
rho = 510.0      # 密度 [kg/m³]
cp = 1380.0        # 比熱容量 [J/(kg·K)]
alpha = k / (rho * cp)  # 熱拡散率 [m²/s]
# 左右壁表面の対流境界条件
h1 = 30.0         # 左表面の熱伝達率 [W/(m²·K)]
h2 = 60.0         # 右表面の熱伝達率 [W/(m²·K)]

# 空間と時間の分割
Nx = 60           # 空間分割数
dx = L / Nx       # 空間格子間隔 [m]
x = np.linspace(0, L, Nx + 1)  # 左端(0)から右端(L)までの座標配列

# 安定条件を考慮して時間刻みを決定
dt = (dx**2) / (2.5 * alpha) 
total_time = 3600 # 計算する全時間 [秒] (1時間)
Nt = int(total_time / dt)

# 観測したい「左表面からの深さ [cm]」を5点指定
target_depths_cm = [0.0, 1.5, 4.5, 7.5, 9.0]  # 0cm(左表面) 〜 9cm(右表面)

# 指定された深さに最も近い格子のインデックスを検索
target_indices = []
actual_depths_cm = []

for d in target_depths_cm:
    target_x = d / 100.0
    idx = np.argmin(np.abs(x - target_x))
    target_indices.append(idx)
    actual_depths_cm.append(x[idx] * 100.0)

# 各観測点の温度履歴を保存する辞書と時間履歴リスト
temperature_history = {idx: [] for idx in target_indices}
time_history = []

# 温度配列の初期化
T = np.full(Nx + 1, T_init)
T_new = T.copy()

# ==========================================
# 2. タイムステップを進めるループ（差分法）
# ==========================================
current_time = 0.0

for step in range(1, Nt + 1):
    current_time += dt
    
    # ① 内部ノードの計算 (i = 1 から Nx-1)
    for i in range(1, Nx):
        d2T_dx2 = (T[i+1] - 2*T[i] + T[i-1]) / (dx**2)
        T_new[i] = T[i] + alpha * dt * d2T_dx2
        
    # ② 左表面 (x = 0, i = 0) -> 対流境界条件 (h1 = 30)
    # 数式: -k * dT/dx = h1 * (T_env_left - T)
    T_new[0] = T[0] + alpha * dt * (2.0 * T[1] - 2.0 * T[0] + (2.0 * dx * h1 * (T_env_left - T[0]) / k)) / (dx**2)
    
    # ③ 右表面 (x = L, i = Nx) -> 対流境界条件 (h2 = 60)
    # 数式: -k * dT/dx = h2 * (T - T_env_right)
    T_new[Nx] = T[Nx] + alpha * dt * (2.0 * T[Nx-1] - 2.0 * T[Nx] + (2.0 * dx * h2 * (T_env_right - T[Nx]) / k)) / (dx**2)
    
    # 配列の更新
    T = T_new.copy()
    
    # 10ステップに1回、指定した位置の温度を記録
    if step % 10 == 0 or step == Nt:
        time_history.append(current_time)
        for idx in target_indices:
            temperature_history[idx].append(T[idx])

# ==========================================
# 3. 結果の可視化（時間変化プロット）
# ==========================================
plt.figure(figsize=(11, 6))

# 各位置の温度変化をプロット
for idx, depth in zip(target_indices, actual_depths_cm):
    if idx == 0:
        label_text = f"Left Surface (0.0 cm) [h1={h1}]"
    elif idx == Nx:
        label_text = f"Right Surface (3.0 cm) [h2={h2}]"
    else:
        label_text = f"Depth: {depth:.2f} cm"
        
    plt.plot(time_history, temperature_history[idx], label=label_text, linewidth=2)

plt.title("Temperature Transient Response (h1=30 at Left, h2=60 at Right)", fontsize=14)
plt.xlabel("Time [seconds]", fontsize=12)
plt.ylabel("Temperature [°C]", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper right", fontsize=11)
plt.xlim(0, total_time)

plt.show()