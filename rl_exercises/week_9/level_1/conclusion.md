# Week 9 – Model-Based RL  
## Level 1: Core Dyna-PPO vs PPO

---

### 1.1 Sample Efficiency

**Experiment Setup:**  
We trained PPO (`use_model=False`) and Dyna-PPO (`use_model=True`) under the same real environment steps budget (15k). The average return was plotted against the number of real steps.

**Figure:**  
`compare_return.png`

**(a) How many steps sooner does Dyna-PPO reach 80% of PPO's final return?**  
PPO's final return peaks around 100–110.  
Dyna-PPO first reaches 80% of that (~80 return) at around 10000 steps, while PPO takes around 14000–15000.  
→ Dyna-PPO achieves this performance roughly 4000–5000 steps earlier.

**(b) Is there an initial "model learning penalty" (Dyna underperforms early)?**  
Yes. In the first 3000–4000 steps, Dyna-PPO consistently underperforms PPO. This is due to inaccurate early model predictions harming policy updates.

---

### 1.2 Model Prediction Accuracy

**Figures:**  
- `dyna_model_loss.png`  
- `dyna_imag_loss.png`

**(a) Around what MSE threshold does Dyna-PPO begin to outperform PPO?**  
State and reward MSE drop below 0.01 within the first 1000 steps, and stabilize around 0.001 after 5000–6000 steps.  
This corresponds to the point where Dyna-PPO starts outperforming PPO.  
→ Dyna begins to outperform PPO when model MSE ≈ 0.005 or lower.

**(b) How fast does the error grow, and how does that inform `imag_horizon`?**  
Once converged, the model MSE remains low and stable.  
This suggests that increasing `imag_horizon` is safe and may improve performance further.

---

### Summary

| Metric               | PPO         | Dyna-PPO        |
|----------------------|-------------|-----------------|
| Early performance    | Better      | Worse initially |
| Sample efficiency    | Slower      | Faster (~5k)    |
| Stability            | High        | Initially low   |
| Model prediction MSE | Not used    | Converges < 0.001 |
