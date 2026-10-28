"""
=============================================================================
Project 02: Logistic Regression Binary Classifier in PyTorch
=============================================================================
Author     : Gideon Mwiti
Challenge  : 100 PyTorch Projects in 10 Days
Topic      : Binary Classification, Sigmoid Activation, Binary Cross-Entropy (BCE)
Goal       : Predict binary labels (0 or 1) using y_hat = sigmoid(w * x + b)
=============================================================================
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def generate_2d_classification_data(num_samples_per_class: int = 150):
    """
    Generates a 2D linearly separable dataset for binary classification:
    - Class 0: Centered around (-1.5, -1.5)
    - Class 1: Centered around (1.5, 1.5)
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Class 0: Mean [-1.5, -1.5], cov = 0.8
    mean_0 = torch.tensor([-1.5, -1.5])
    x_class0 = torch.randn(num_samples_per_class, 2) * 0.85 + mean_0
    y_class0 = torch.zeros(num_samples_per_class, 1)

    # Class 1: Mean [1.5, 1.5], cov = 0.8
    mean_1 = torch.tensor([1.5, 1.5])
    x_class1 = torch.randn(num_samples_per_class, 2) * 0.85 + mean_1
    y_class1 = torch.ones(num_samples_per_class, 1)

    # Combine into single dataset
    x = torch.cat([x_class0, x_class1], dim=0)
    y = torch.cat([y_class0, y_class1], dim=0)

    # Shuffle data
    indices = torch.randperm(x.size(0))
    x = x[indices]
    y = y[indices]

    return x, y

class LogisticRegressionModel(nn.Module):
    """
    Logistic Regression Model for 2D inputs:
    z = x1 * w1 + x2 * w2 + b
    y_hat = sigmoid(z) = 1 / (1 + exp(-z))
    """
    def __init__(self, input_dim: int = 2, output_dim: int = 1):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute logit z and apply sigmoid non-linearity
        return torch.sigmoid(self.linear(x))

def compute_metrics(y_true: torch.Tensor, y_pred_probs: torch.Tensor, threshold: float = 0.5):
    """
    Computes Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
    """
    preds = (y_pred_probs >= threshold).float()
    
    tp = ((preds == 1) & (y_true == 1)).sum().item()
    tn = ((preds == 0) & (y_true == 0)).sum().item()
    fp = ((preds == 1) & (y_true == 0)).sum().item()
    fn = ((preds == 0) & (y_true == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    confusion_matrix = np.array([[int(tn), int(fp)], [int(fn), int(tp)]])
    return accuracy, precision, recall, f1, confusion_matrix

def train_logistic_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 150,
    lr: float = 0.1
):
    """
    Trains the logistic regression classifier using Binary Cross-Entropy loss.
    """
    model = LogisticRegressionModel(input_dim=2, output_dim=1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print("=================================================================")
    print("      PROJECT 02: LOGISTIC REGRESSION CLASSIFIER (PYTORCH)       ")
    print("=================================================================")
    print(f"Dataset Size       : {x.size(0)} samples (2D Features -> Binary Label)")
    print(f"Hyperparameters    : Epochs = {epochs}, Learning Rate = {lr}")
    print("-----------------------------------------------------------------")
    print(f"{'Epoch':^8} | {'BCE Loss':^12} | {'Accuracy (%)':^14} | {'F1-Score':^12}")
    print("-----------------------------------------------------------------")

    history = {
        "epoch": [],
        "loss": [],
        "accuracy": []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        
        # Forward pass: compute predicted probabilities
        y_pred = model(x)
        
        # Compute Binary Cross Entropy Loss
        loss = criterion(y_pred, y)

        # Backward pass: compute gradients
        optimizer.zero_grad()
        loss.backward()

        # Parameter update
        optimizer.step()

        # Compute evaluation metrics
        with torch.no_grad():
            acc, prec, rec, f1, cm = compute_metrics(y, y_pred)

        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["accuracy"].append(acc * 100)

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f"{epoch:^8d} | {loss.item():^12.5f} | {acc * 100:^14.2f} | {f1:^12.4f}")

    print("-----------------------------------------------------------------")
    w = model.linear.weight.data.squeeze().tolist()
    b = model.linear.bias.data.item()
    print(f"Learned Weights    : w1 = {w[0]:.4f}, w2 = {w[1]:.4f}")
    print(f"Learned Bias       : b  = {b:.4f}")
    print(f"Final Accuracy     : {acc * 100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("=================================================================\n")

    return model, history, (acc, prec, rec, f1, cm)

def plot_results(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    history: dict,
    metrics: tuple,
    output_path: str
):
    """
    Renders a comprehensive 4-panel diagnostic graphic:
    1. 2D Decision Boundary with Probability Heatmap
    2. BCE Loss Convergence Curve
    3. Accuracy Progression Curve
    4. Annotated Confusion Matrix
    """
    acc, prec, rec, f1, cm = metrics
    x_np = x.numpy()
    y_np = y.numpy().squeeze()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Project 02: PyTorch Logistic Regression Classifier (2D Binary Decision Space)\n"
        f"Final Accuracy = {acc*100:.2f}% | F1-Score = {f1:.4f} | Precision = {prec:.4f} | Recall = {rec:.4f}",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )

    # 1. Decision Boundary & Probability Landscape
    ax1 = axes[0, 0]
    x_min, x_max = x_np[:, 0].min() - 1.0, x_np[:, 0].max() + 1.0
    y_min, y_max = x_np[:, 1].min() - 1.0, x_np[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        probs = model(grid_tensor).reshape(xx.shape).numpy()

    # Probability contour background
    contour = ax1.contourf(xx, yy, probs, levels=25, cmap="RdBu", alpha=0.7)
    cbar = plt.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("P(y = 1 | x)", fontsize=10)

    # Decision boundary contour at P = 0.5
    ax1.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2.5, linestyles="--")

    # Scatter points by class
    ax1.scatter(x_np[y_np == 0, 0], x_np[y_np == 0, 1], color="#DC2626", edgecolors="black", linewidth=0.5, label="Class 0 (Negative)", alpha=0.9)
    ax1.scatter(x_np[y_np == 1, 0], x_np[y_np == 1, 1], color="#2563EB", edgecolors="black", linewidth=0.5, label="Class 1 (Positive)", alpha=0.9)
    ax1.set_title("1. Decision Boundary & Probability Contours (P=0.5)", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Feature 1 ($x_1$)")
    ax1.set_ylabel("Feature 2 ($x_2$)")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # 2. Binary Cross-Entropy Loss Curve
    ax2 = axes[0, 1]
    ax2.plot(history["epoch"], history["loss"], color="#DC2626", linewidth=2.2)
    ax2.set_title("2. Binary Cross-Entropy (BCE) Loss Convergence", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("BCE Loss")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.text(
        0.95, 0.90,
        f"Initial Loss: {history['loss'][0]:.4f}\nFinal Loss: {history['loss'][-1]:.4f}",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#DC2626", alpha=0.9)
    )

    # 3. Accuracy Progression Curve
    ax3 = axes[1, 0]
    ax3.plot(history["epoch"], history["accuracy"], color="#10B981", linewidth=2.2)
    ax3.set_title("3. Classification Accuracy Progression (%)", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_ylim([0, 105])
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.text(
        0.95, 0.15,
        f"Peak Accuracy: {max(history['accuracy']):.2f}%",
        transform=ax3.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#10B981", alpha=0.9)
    )

    # 4. Confusion Matrix Heatmap
    ax4 = axes[1, 1]
    im = ax4.imshow(cm, interpolation="nearest", cmap="Blues")
    ax4.set_title("4. Test Confusion Matrix", fontweight="bold", fontsize=11)
    tick_marks = np.arange(2)
    ax4.set_xticks(tick_marks)
    ax4.set_yticks(tick_marks)
    ax4.set_xticklabels(["Predicted Class 0", "Predicted Class 1"])
    ax4.set_yticklabels(["Actual Class 0", "Actual Class 1"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            label = f"{val}\n({'TN' if (i==0 and j==0) else 'FP' if (i==0 and j==1) else 'FN' if (i==1 and j==0) else 'TP'})"
            ax4.text(j, i, label, ha="center", va="center",
                     color="white" if val > thresh else "black",
                     fontweight="bold", fontsize=11)
    ax4.set_ylabel("True Ground Truth")
    ax4.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Diagnostic plot saved successfully at: {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_plot_path = os.path.join(script_dir, "classification_results.png")

    EPOCHS = 150
    LEARNING_RATE = 0.1

    # Step 1: Generate 2D classification dataset
    x, y = generate_2d_classification_data(num_samples_per_class=150)

    # Step 2: Train logistic regression classifier
    model, history, metrics = train_logistic_regression(
        x=x,
        y=y,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )

    # Step 3: Render and save results visualization
    plot_results(
        x=x,
        y=y,
        model=model,
        history=history,
        metrics=metrics,
        output_path=output_plot_path
    )

if __name__ == "__main__":
    main()

