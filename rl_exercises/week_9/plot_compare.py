import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 路径设置
ppo_path = "rl_exercises/week_9/level_1/ppo/training_log.csv"
dyna_path = "rl_exercises/week_9/level_1/dyna_ppo/training_log.csv"

# 读取数据
ppo_df = pd.read_csv(ppo_path)
ppo_df["method"] = "PPO"

dyna_df = pd.read_csv(dyna_path)
dyna_df["method"] = "Dyna-PPO"

# 合并数据
df = pd.concat([ppo_df, dyna_df])

# ===== 图 1：Return vs Real Steps =====
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="real_steps", y="return", hue="method")
plt.title("Sample Efficiency: Return vs Real Env Steps")
plt.xlabel("Real Env Steps")
plt.ylabel("Average Return")
plt.grid()
plt.tight_layout()
plt.savefig("compare_return.png")
print("✅ compare_return.png saved.")

# ===== 图 2：Dyna 专属 - 模型精度 =====
plt.figure(figsize=(10, 5))
sns.lineplot(data=dyna_df, x="real_steps", y="model_s_loss", label="State MSE")
sns.lineplot(data=dyna_df, x="real_steps", y="model_r_loss", label="Reward MSE")
plt.title("Dyna-PPO Model Prediction Loss")
plt.xlabel("Real Env Steps")
plt.ylabel("MSE Loss")
plt.grid()
plt.tight_layout()
plt.savefig("dyna_model_loss.png")
print("✅ dyna_model_loss.png saved.")

# ===== 图 3：Dyna 专属 - Imagined PPO loss =====
plt.figure(figsize=(10, 5))
sns.lineplot(data=dyna_df, x="real_steps", y="imag_p_loss", label="Imag Policy Loss")
sns.lineplot(data=dyna_df, x="real_steps", y="imag_v_loss", label="Imag Value Loss")
sns.lineplot(data=dyna_df, x="real_steps", y="imag_e_loss", label="Imag Entropy Loss")
plt.title("Dyna-PPO Imagined PPO Losses")
plt.xlabel("Real Env Steps")
plt.ylabel("Loss")
plt.grid()
plt.tight_layout()
plt.savefig("dyna_imag_loss.png")
print("✅ dyna_imag_loss.png saved.")
