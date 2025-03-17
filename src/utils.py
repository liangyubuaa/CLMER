import torch
import os
from src.dataset import Multimodal_Datasets
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from torch import nn

def get_data(hyp_params, dataset, split='train'):
    data_path = os.path.join(hyp_params.data_folder,dataset) + f'_{split}.dt'
    print(data_path)
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(hyp_params.data_folder, dataset, split)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data



def save_model(model, name=''):
    torch.save(model, f'pre_trained_models/{name}.pt')


def load_model(name=''):
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model



def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def loop_iterable(iterable):
    while True:
        yield from iterable


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


from shutil import copy


def copy_Files(destination, data_type, modality):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("clmain.py", os.path.join(destination_dir, "clmain.py"))
    # copy("args.py", os.path.join(destination_dir, "args.py"))
    copy("src/cltrain.py", os.path.join(destination_dir, "cltrain.py"))
    copy(f"config_files/{data_type[:-1]+modality}_Configs.py", os.path.join(destination_dir, f"{data_type[:-1]+modality}_Configs.py"))
    copy("dataloader/augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/encoder.py", os.path.join(destination_dir, f"encoder.py"))
    copy("models/loss.py", os.path.join(destination_dir, "loss.py"))
    copy("models/timeseries.py", os.path.join(destination_dir, "timeseries.py"))
