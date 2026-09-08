import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

from dataloader.dataloader import data_generator
from models.timeseries import timeseries
from models.encoder import base_Model, SampleProjector
from src.cltrain import Trainer, model_evaluate, gen_pseudo_labels
from src.utils import _calc_metrics, copy_Files
from src.utils import _logger, set_requires_grad

start_time = datetime.now()

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description',     default='default',  type=str,   help='Experiment Description')
parser.add_argument('--description',            default='default',            type=str,   help='Experiment Description')
parser.add_argument('--seed',                       default=0,                  type=int,   help='seed value')
parser.add_argument('--training_mode',              default='self_supervised',  type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, SupCon, ft_1p, gen_pseudo_labels')
parser.add_argument('--withtl',           default=0,              type=int)

parser.add_argument('--selected_dataset',           default='default',              type=str,   help='Dataset of choice: EEG, HAR, Epilepsy, pFD')
parser.add_argument('--data_path',                  default=r'data/',           type=str,   help='Path containing dataset')
parser.add_argument('--modality',           default='physio',              type=str)
parser.add_argument('--pairwise',           default=1,              type=int)

parser.add_argument('--logs_save_dir',              default='experiments_logs', type=str,   help='saving directory')
parser.add_argument('--device',                     default='cuda:0',           type=str,   help='cpu or cuda')
parser.add_argument('--home_path',                  default=home_dir,           type=str,   help='Project home directory')
parser.add_argument('--sample_loss_mode', default='default', type=str,
                    choices=['default', 'sample'],
                    help='sample loss branch: "default" = context loss, "sample" = SimCLR sample-level loss')
args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
training_mode = args.training_mode
run_description = args.description
modality = args.modality
pairwise = args.pairwise
withtl = args.withtl

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

exec(f'from config_files.{data_type[:-1]+modality}_Configs import Config as Configs')
configs = Configs()
# command-line override for the sample loss branch (read via getattr elsewhere)
configs.sample_loss_mode = args.sample_loss_mode

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################


experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, modality,
                                  training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
data_path = os.path.join(args.data_path, data_type)
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode, data_type, modality)
logger.debug("Data loaded ...")
ft_perc = "25p"
# Load Model
model = base_Model(configs).to(device)
temporal_contr_model = timeseries(configs, device).to(device)
# sample-level projection head; only created/used when sample_loss_mode == "sample".
# In the default "context" mode it stays None so the original path is unchanged.
if getattr(configs, "sample_loss_mode", "default") == "sample":
    sample_projector = SampleProjector(configs).to(device)
else:
    sample_projector = None
if "fine_tune" in training_mode or "ft_" in training_mode:
    # load saved model of this experiment
    if 'SupCon' not in training_mode:
        if withtl == 0:
            load_from = os.path.join(
                os.path.join(logs_save_dir, experiment_description, run_description, f"{modality}/self_supervised_seed_{SEED}",
                             "saved_models"))
        else:
            load_from = os.path.join(
                os.path.join(logs_save_dir, experiment_description, run_description, f"{modality}/train_seed_{SEED}",
                             "saved_models_with_tl"))
    else:
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"{modality}/SupCon_seed_{SEED}", "saved_models"))
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

if training_mode == "gen_pseudo_labels":
    load_from = os.path.join(
        os.path.join(logs_save_dir, experiment_description, run_description, f"{modality}/fine_tune_{ft_perc}_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model.load_state_dict(pretrained_dict)
    gen_pseudo_labels(model, train_dl, device, data_path, modality)
    sys.exit(0)

if "train_linear" in training_mode or "selfpairwise" in training_mode:
    if 'SupCon' not in training_mode:
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"{modality}/self_supervised_seed_{SEED}",
                         "saved_models"))
    else:
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"{modality}/SupCon_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # delete these parameters (Ex: the linear layer at the end)
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == "random_init":
    model_dict = model.state_dict()

    # delete all the parameters except for logits
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.


if training_mode == "SupCon":
    load_from = os.path.join(
        os.path.join(logs_save_dir, experiment_description, run_description, f"{modality}/fine_tune_{ft_perc}_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model.load_state_dict(pretrained_dict)


# model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
#                                    weight_decay=3e-4)
model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=3e-4)
temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, weight_decay=3e-4)
if sample_projector is not None:
    temporal_contr_optimizer.add_param_group({'params': sample_projector.parameters()})
# temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
#                                             betas=(configs.beta1, configs.beta2))
if training_mode == "self_supervised" or training_mode == "SupCon":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type, modality)
print(temporal_contr_optimizer.param_groups[0]['betas'])
# Trainer
Trainer(data_path, configs, data_type, model, temporal_contr_model, sample_projector, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device,
        logger, configs, experiment_log_dir, training_mode, modality, pairwise)

if training_mode != "self_supervised" and training_mode != "SupCon" and training_mode != "SupCon_pseudo":
    # Testing
    outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
    total_loss, total_acc, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Training time is : {datetime.now() - start_time}")
