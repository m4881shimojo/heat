import numpy as np
import matplotlib.pyplot as plt
#20260602 shimojo &gemini
# ==========================================
# 1. 物性値と計算条件の設定
# ==========================================
R = 0.06          # 鉄球の半径 [m] (5cm)
T_init = 20.0     # 鉄球の初期温度 [°C]
T_env = 200.0     # 周囲の温度 [°C]

# 鉄の物性値
k = 50.0          # 熱伝導率 [W/(m·K)]
rho = 7800.0      # 密度 [kg/m³]
cp = 460.0        # 比熱容量 [J/(kg·K)]
alpha = k / (rho * cp)  # 熱拡散率 [m²/s]

# 表面の熱伝達率
h = 100.0         # 熱伝達率 [W/(m²·K)]

# 空間と時間の分割
Nx = 100          # 空間分割数（精度向上のため少し細かくしています）
dx = R / Nx       # 空間格子間隔 [m]
r = np.linspace(0, R, Nx + 1)  # 中心(0)から表面(R)までの座標配列

# 安定条件を考慮して時間刻みを決定
dt = (dx**2) / (4.0 * alpha) 
total_time = 600  # 計算する全時間 [秒] (10分)
#total_time = 2000  # 計算する全時間 [秒] (10分)
Nt = int(total_time / dt)

# ------------------------------------------
# 【新規】観測したい「表面からの深さ [cm]」を5点指定
# ------------------------------------------
#target_depths_cm = [0.0, 1.0, 2.0, 3.0, 5.0]  # 0cm(表面) 〜 5cm(中心)
target_depths_cm = [0.0,1.0, 3.0, 5.0]  # 0cm(表面) 〜 5cm(中心)

# 深さを「中心からの半径 r [m]」に変換し、最も近い格子のインデックスを検索
target_indices = []
actual_depths_cm = []

for d in target_depths_cm:
    target_r = R - (d / 100.0)  # 半径 = 全体半径 - 深さ
    idx = np.argmin(np.abs(r - target_r)) # 最も近い格子の番号
    target_indices.append(idx)
    actual_depths_cm.append((R - r[idx]) * 100.0) # 実際に計算される正確な深さ

# 各観測点の温度履歴を保存する辞書
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
    
    # 内部ノードの計算
    for i in range(1, Nx):
        d2T_dr2 = (T[i+1] - 2*T[i] + T[i-1]) / (dx**2)
        dT_dr = (T[i+1] - T[i-1]) / (2*dx)
        T_new[i] = T[i] + alpha * dt * (d2T_dr2 + (2.0 / r[i]) * dT_dr)
        
    # 境界条件1: 中心 (r = 0, i = 0)
    T_new[0] = T[0] + 3.0 * alpha * dt * (2.0 * (T[1] - T[0]) / (dx**2))
    
    # 境界条件2: 表面 (r = R, i = Nx)
    T_new[Nx] = (k * T_new[Nx-1] + h * dx * T_env) / (k + h * dx)
    
    # 配列の更新
    T = T_new.copy()
    
    # 【新規】1秒ごと（または全ステップ）に指定した点の温度を記録
    # 計算軽量化のため、10ステップに1回記録
    if step % 10 == 0 or step == Nt:
        time_history.append(current_time)
        for idx in target_indices:
            temperature_history[idx].append(T[idx])

# ==========================================
# 3. 結果の可視化（時間変化プロット）
# ==========================================
plt.figure(figsize=(10, 6))

# 各深さの温度変化をプロット
for idx, depth in zip(target_indices, actual_depths_cm):
    # 中心（半径0）の場合は「中心」とラベル表示
    label_text = "Center (5.0 cm)" if idx == 0 else f"Depth: {depth:.1f} cm"
    plt.plot(time_history, temperature_history[idx], label=label_text, linewidth=2)

plt.title("Temperature Transient Response at Different Depths", fontsize=14)
plt.xlabel("Time [seconds]", fontsize=12)
plt.ylabel("Temperature [°C]", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower right", fontsize=11)
plt.xlim(0, total_time)
plt.ylim(T_init - 10, T_env + 10)
#plt.ylim(20, 200)

plt.show()