import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_training_results(checkpoint_path, save_plot=False, plot_name=None):
    """
    绘制训练/验证 Loss 和验证 Accuracy 曲线。
    
    参数:
        checkpoint_path (str): 指向 checkpoint 文件夹路径（包含 trainer_state.json）。
        save_plot (bool): 是否保存为 PDF 文件，默认 False。
        plot_name (str): 保存文件名（不含扩展名），默认 None，则根据 checkpoint 自动命名。
    """

    trainer_state_file = os.path.join(checkpoint_path, "trainer_state.json")
    with open(trainer_state_file, 'r') as f:
        trainer_state = json.load(f)

    log_history = trainer_state["log_history"]

    train_data = [(entry["step"], entry["loss"], 'Train Loss') for entry in log_history if "loss" in entry]
    eval_data = [(entry["step"], entry["eval_loss"], 'Validation Loss') for entry in log_history if "eval_loss" in entry]
    eval_acc_data = [(entry["step"], entry["eval_accuracy"]) for entry in log_history if "eval_accuracy" in entry]

    loss_data = pd.DataFrame(train_data + eval_data, columns=["Step", "Loss", "Type"])
    acc_data = pd.DataFrame(eval_acc_data, columns=["Step", "Accuracy"])

    sns.set(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=loss_data,
        x="Step",
        y="Loss",
        hue="Type",
        palette={"Train Loss": "#3498db", "Validation Loss": "#f5b041"},
        linewidth=1.8,
        ax=ax1
    )

    ax1.set_yscale('log')
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, which="both", ls="--", linewidth=0.5)

    ax1.set_xlim(0, loss_data["Step"].max() + 200)
    xticks = ax1.get_xticks()
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{int(x/1000)}k" for x in xticks])

    ax2 = ax1.twinx()
    sns.lineplot(
        data=acc_data,
        x="Step",
        y="Accuracy",
        color="#e74c3c",
        linewidth=1.8,
        ax=ax2,
        label="Validation Accuracy"
    )
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_ylim(0.65, 1.0)

    ax2.set_yticks([i/100 for i in range(65, 101, 5)])
    ax2.set_yticklabels([f"{i}%" for i in range(65, 101, 5)])

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    fig.legend(
        handles, 
        labels, 
        loc='upper right', 
        bbox_to_anchor=(0.90, 0.85), 
        fontsize=11, 
        title="Metrics", 
        title_fontsize=12, 
        frameon=True,  
        shadow=False
    )

    ax1.legend_.remove()
    ax2.legend_.remove()


    plt.title("Training Loss, Validation Loss, and Accuracy Curve", fontsize=16)
    plt.tight_layout()

    if save_plot:
        os.makedirs('./plot', exist_ok=True)
        if plot_name is None:
            checkpoint_folder = os.path.basename(os.path.normpath(checkpoint_path))
            plot_name = f"{checkpoint_folder}_plot"
        save_path = os.path.join('./plot', f"{plot_name}.pdf")
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()
