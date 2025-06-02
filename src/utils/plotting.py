import seaborn as sns
from matplotlib import pyplot as plt


def plot_log_info(config_info, mode):
    if mode == "segmentation":
        train_model = config_info["trained_model"]
        encoder = config_info["encoder"]
        classes = config_info["classes"] # list of strs
        classes = ", ".join(classes)
        patch_size = config_info["patch_size"]
        log_file = config_info["logs"]
        title_suffix = f"model: {train_model}, encoder: {encoder}, classes: ({classes}), patch_size: {patch_size}"


        fig, ax = plt.subplots(3, 1, figsize=(20, 10 * 3))
        sns.lineplot(data=log_file, x="epoch", y="train_loss", ax=ax[0], label="train_loss")
        sns.lineplot(data=log_file, x="epoch", y="val_loss", ax=ax[0], label="val_loss")
        ax[0].set_title("Loss, " + title_suffix)

        sns.lineplot(data=log_file, x="epoch", y="train_mean_iou", ax=ax[1], label="train_mean_iou")
        sns.lineplot(data=log_file, x="epoch", y="val_mean_iou", ax=ax[1], label="val_mean_iou")
        ax[1].set_title("Mean IoU, " + title_suffix)

        sns.lineplot(data=log_file, x="epoch", y="train_mean_dice", ax=ax[2], label="train_mean_dice")
        sns.lineplot(data=log_file, x="epoch", y="val_mean_dice", ax=ax[2], label="val_mean_dice")
        ax[2].set_title("Mean Dice, " + title_suffix)
    elif mode == "cnn":
        encoder = config_info["encoder"]
        patch_size = config_info["patch_size"]
        log_file = config_info["logs"]
        title_suffix = f"model: {encoder}, patch_size: {patch_size}"

        fig, ax = plt.subplots(3, 1, figsize=(20, 10 * 2))
        sns.lineplot(data=log_file, x="epoch", y="train_loss", ax=ax[0], label="train_loss")
        sns.lineplot(data=log_file, x="epoch", y="val_loss", ax=ax[0], label="val_loss")
        ax[0].set_title("Loss, " + title_suffix)

        sns.lineplot(data=log_file, x="epoch", y="avg_train_accuracy", ax=ax[1], label="train_accuracy")
        sns.lineplot(data=log_file, x="epoch", y="avg_val_accuracy", ax=ax[1], label="val_accuracy")
        ax[1].set_title("Accuracy, " + title_suffix)

        sns.lineplot(data=log_file, x="epoch", y="avg_train_f1", ax=ax[2], label="train_f1")
        sns.lineplot(data=log_file, x="epoch", y="avg_val_f1", ax=ax[2], label="val_f1")
        ax[2].set_title("F1, " + title_suffix)

        model_name = config_info["model_path"].split("/")[-1]
        config_name_list = model_name.split("_")[0:2]
        config_name = " ".join(config_name_list)
        fig.suptitle(config_name, fontsize=18, weight="bold")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return fig, ax
