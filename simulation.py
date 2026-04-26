import numpy as np
import matplotlib.pyplot as plt

# 空間
N = 200
x = np.linspace(0, 10, N)
dx = x[1] - x[0]

# 時間
dt = 0.01
steps = 500

# 意味場 M（適当に定義）
M = np.sin(x) + 0.3 * np.sin(3 * x)

# 勾配 ∇M
grad_M = np.gradient(M, dx)

# 問い強度 Q
Q = np.abs(grad_M)

# 閾値
theta = 0.5

# Γ（構造生成）
def Gamma(Q):
    return np.maximum(Q - theta, 0)

# 共鳴 R（簡易）
R = np.ones_like(x) * 0.8

# 構造 S 初期値
S = np.zeros_like(x)

# 保存用
history = []

# 時間発展
for t in range(steps):
    dSdt = Gamma(Q) * R
    S = S + dt * dSdt
    history.append(S.copy())

history = np.array(history)

# 可視化
plt.figure(figsize=(10,6))

# 最初と最後
plt.plot(x, history[0], label="Initial")
plt.plot(x, history[-1], label="Final")

plt.title("I2OS Structure Evolution Simulation")
plt.xlabel("x")
plt.ylabel("S(x)")
plt.legend()
plt.grid()

plt.show()