import seaborn as sns
from matplotlib import pyplot as plt


def plot_log_info(config_info):
    train_model = config_info["trained_model"]
    encoder = config_info["encoder"]
    classes = config_info["classes"] # list of strs
    classes = ", ".join(classes)
    patch_size = config_info["patch_size"]
    log_file = config_info["logs"]


    fig, ax = plt.subplots(3, 1, figsize=(20, 10 * 3))
    sns.lineplot(data=log_file, x="epoch", y="train_loss", ax=ax[0], label="train_loss")
    sns.lineplot(data=log_file, x="epoch", y="val_loss", ax=ax[0], label="val_loss")
    ax[0].set_title(f"Loss, model: {train_model}, encoder: {encoder}, classes: ({classes}), patch_size: {patch_size}")

    sns.lineplot(data=log_file, x="epoch", y="train_mean_iou", ax=ax[1], label="train_mean_iou")
    sns.lineplot(data=log_file, x="epoch", y="val_mean_iou", ax=ax[1], label="val_mean_iou")
    ax[1].set_title(f"Mean IoU, model: {train_model}, encoder: {encoder}, classes: ({classes}), patch_size: {patch_size}")

    sns.lineplot(data=log_file, x="epoch", y="train_mean_dice", ax=ax[2], label="train_mean_dice")
    sns.lineplot(data=log_file, x="epoch", y="val_mean_dice", ax=ax[2], label="val_mean_dice")
    ax[2].set_title(f"Mean Dice, model: {train_model}, encoder: {encoder}, classes: ({classes}), patch_size: {patch_size}")
    return fig, ax
