import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

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



def plot_confusion_matrix(
    trainer, 
    dataset, 
    label_names=None, 
    normalize=None, 
    cmap=plt.cm.Blues, 
    save_name=None
):
    """
    常用颜色风格（cmap）包括：
    - plt.cm.Blues       蓝色（默认）
    - plt.cm.Greens      绿色
    - plt.cm.Oranges     橙色
    - plt.cm.Reds        红色
    - plt.cm.Purples     紫色
    - plt.cm.Greys       灰度
    - plt.cm.cividis     现代感配色
    - plt.cm.plasma      炫彩感配色
    - plt.cm.inferno     深红-黄
    """

    label_mapping = {
        0: 5, 1: 2, 2: 3, 3: 8, 4: 0, 5: 7, 6: 4, 7: 1, 8: 6
    }

    def map_label(example):
        example["label"] = label_mapping[example["label"]]
        return example

    revised_dataset = dataset.map(map_label)

    if label_names is None:
        inverse_mapping = {v: k for k, v in label_mapping.items()}
        label_names = [str(inverse_mapping[i]) for i in range(len(inverse_mapping))]

    if save_name != None:
        save_dir = "./confusion_matrix"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name if save_name.endswith(".pdf") else save_name + ".pdf")

    predictions = trainer.predict(revised_dataset)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=cmap, ax=ax, values_format=".2f" if normalize else "d")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.grid(False)
    plt.tight_layout()

    if save_name != None:
        plt.savefig(save_path, format="pdf")
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.show()