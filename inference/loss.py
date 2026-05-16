import ast
import matplotlib.pyplot as plt

log_path = "output1027.log"

losses, grad_norms, lrs, steps = [], [], [], []

with open(log_path, "r", encoding="utf-8") as f:
    for step, line in enumerate(f, start=1):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            record = ast.literal_eval(line)
        except Exception:
            continue

        if not all(k in record for k in ("loss", "grad_norm", "learning_rate")):
            continue

        losses.append(record["loss"])
        grad_norms.append(record["grad_norm"])
        lrs.append(record["learning_rate"])
        steps.append(len(losses))


plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(steps, losses, label="loss", color="tab:blue")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(steps, grad_norms, label="grad_norm", color="tab:orange")
plt.xlabel("Step")
plt.ylabel("Grad Norm")
plt.title("Gradient Norm")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(steps, lrs, label="learning_rate", color="tab:green")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate")
plt.grid(True)

plt.tight_layout()
plt.savefig("train_metrics_1027.png", dpi=300)
