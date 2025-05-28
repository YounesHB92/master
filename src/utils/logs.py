import csv

def init_cnn_logfile(log_path, class_names):
    header = [
        "epoch", "train_loss", "val_loss",
        "avg_train_f1", "avg_val_f1",
        "avg_train_accuracy", "avg_val_accuracy"
    ]

    # Add per-class F1 and Accuracy
    for cls in class_names:
        header.append(f"train_f1_{cls}")
        header.append(f"train_acc_{cls}")
    for cls in class_names:
        header.append(f"val_f1_{cls}")
        header.append(f"val_acc_{cls}")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

def append_cnn_logfile(log_path, row_data):
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row_data)