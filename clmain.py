import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

from dataloader.dataloader import data_generator
from models.timeseries import timeseries
from models.encoder import base_Model
from src.cltrain import Trainer, model_evaluate
from src.utils import _calc_metrics, copy_Files
from src.utils import _logger, set_requires_grad

start_time = datetime.now()

parser = argparse.ArgumentParser()

home_dir = os.getcwd()
parser.add_argument('--experiment', default='amigos4class0', type=str, help='Experiment')
parser.add_argument('--description', default='amigos4class0', type=str,   help='Description')
parser.add_argument('--seed', default=1111, type=int)
parser.add_argument('--training_mode', default='self_supervised', type=str)
parser.add_argument('--selected_dataset', default='amigos4class0', type=str)
parser.add_argument('--data_path', default=r'data/', type=str)
parser.add_argument('--modality', default='physio', type=str)
parser.add_argument('--logs', default='logs', type=str)
parser.add_argument('--device', default='cuda:0', type=str,)
parser.add_argument('--home_path', default=home_dir, type=str)
args = parser.parse_args()

device = torch.device(args.device)
experiment = args.experiment
data_type = args.selected_dataset
training_mode = args.training_mode
description = args.description
modality = args.modality

logs_save_dir = args.logs
os.makedirs(logs_save_dir, exist_ok=True)

exec(f'from config_files.{data_type[:-1]+modality}_Configs import Config as Configs')
configs = Configs()

SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

experiment_log_dir = os.path.join(logs_save_dir, experiment, description, modality,
                                  training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

counter = 0
src_counter = 0

log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

data_path = os.path.join(args.data_path, data_type)
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode, data_type, modality)
logger.debug("Data loaded ...")

model = base_Model(configs).to(device)
temporal_contr_model = timeseries(configs, device).to(device)
if "fine_tune" in training_mode:
    load_from = os.path.join(
        os.path.join(logs_save_dir, experiment, description, f"{modality}/self_supervised_seed_{SEED}",
                     "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


if "train_linear" in training_mode or "pairwise" in training_mode:
    
    load_from = os.path.join(
    os.path.join(logs_save_dir, experiment, description, f"{modality}/self_supervised_seed_{SEED}",
                 "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)

if training_mode == "random_init":
    model_dict = model.state_dict()

    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(model, model_dict, requires_grad=False) 

# model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
#                                    weight_decay=3e-4)
model_optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=3e-4)
temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=1e-3, weight_decay=3e-4)
# temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
#                                             betas=(configs.beta1, configs.beta2))
if training_mode == "self_supervised" :  
    copy_Files(os.path.join(logs_save_dir, experiment, description), data_type, modality)
print(temporal_contr_optimizer.param_groups[0]['betas'])
Trainer(data_type, model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device,
        logger, configs, experiment_log_dir, training_mode, modality)

if training_mode != "self_supervised":
    outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
    total_loss, total_acc, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Training time is : {datetime.now() - start_time}")
