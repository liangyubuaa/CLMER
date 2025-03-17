import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from transformers import get_scheduler

from models.loss import NTXentLoss
import tensorboard
from torch.utils.tensorboard import SummaryWriter

def Trainer(dataset, model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode, modality):
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, mode='min', verbose=True)
    scheduler1 = get_scheduler(
      "cosine",
      optimizer=model_optimizer,
      num_warmup_steps=20,
      num_training_steps=1000
    )
    scheduler2 = get_scheduler(
      "cosine",
      optimizer=temp_cont_optimizer,
      num_warmup_steps=20,
      num_training_steps=1000
    )
    
    writer = SummaryWriter(log_dir = experiment_log_dir)
    
    if "pairwise" in training_mode:
        feature_generator(dataset, model, train_dl, valid_dl, test_dl, device, training_mode, modality)
        exit()
            
    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                             epoch,
                                            criterion, train_dl, config, device, training_mode, writer)
        writer.add_scalar('Train Loss', train_loss,epoch)
        writer.add_scalar('Train Accuracy', train_acc,epoch)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        scheduler1.step()
        scheduler2.step()
        writer.add_scalar('Learning Rate1', scheduler1.optimizer.param_groups[0]['lr'],epoch)
        writer.add_scalar('Learning Rate2', scheduler2.optimizer.param_groups[0]['lr'],epoch)
        if training_mode != 'self_supervised' and training_mode != 'SupCon':
            scheduler1.step(valid_loss)
            writer.add_scalar('Learning Rate1', scheduler1.optimizer.param_groups[0]['lr'],epoch)
        
        writer.add_scalar('Valid Loss', valid_loss,epoch)
        writer.add_scalar('Valid Accuracy', valid_acc,epoch)
        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if (training_mode != "self_supervised"):
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        writer.add_scalar('Test Loss', test_loss,epoch)
        writer.add_scalar('Test Accuracy', test_acc,epoch)
        logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')
    writer.close()
    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,  epoch, criterion, train_loader, config,
                device, training_mode, writer):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        # print(data)
        # print(labels)
        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(features2, features1)


        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss1 = temp_cont_loss1 + temp_cont_loss2
            loss2 = nt_xent_criterion(temp_cont_feat1, temp_cont_feat2)
            loss = lambda1*loss1+lambda2*loss2 
            # scheduler1.step()
            # writer.add_scalar('Learning Rate1 in Self_Sup', scheduler1.optimizer.param_groups[0]['lr'],epoch)

        else:
            output = model(data)
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc

def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=True)[1]
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode == "self_supervised":
        total_loss = 0
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_loss = torch.tensor(total_loss).mean()
        total_acc = torch.tensor(total_acc).mean()
        return total_loss, total_acc, outs, trgs


def feature_generator(dataset, model, train_loader, valid_loader, test_loader, device, training_mode, modality):
    model.train()
    train_feature_data = torch.Tensor().to(device)
    train_feature_labels = torch.Tensor().to(device)
    valid_feature_data = torch.Tensor().to(device)
    valid_feature_labels = torch.Tensor().to(device)
    test_feature_data = torch.Tensor().to(device)
    test_feature_labels = torch.Tensor().to(device)
    origin = pickle.load(open(f'data/{dataset}/{dataset}.pkl', 'rb'))
    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        output = model(data)
        _, x = output
        train_feature_data = torch.cat((train_feature_data,x), dim=0)
        train_feature_labels = torch.cat((train_feature_labels,labels), dim=0)
    for batch_idx, (data, labels, aug1, aug2) in enumerate(valid_loader):
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        output = model(data)
        _, x = output
        valid_feature_data = torch.cat((valid_feature_data,x), dim=0)
        valid_feature_labels = torch.cat((valid_feature_labels,labels), dim=0)
    for batch_idx, (data, labels, aug1, aug2) in enumerate(test_loader):
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        output = model(data)
        _, x = output
        test_feature_data = torch.cat((test_feature_data,x), dim=0)
        test_feature_labels = torch.cat((test_feature_labels,labels), dim=0)
    print(valid_feature_labels.shape)
    print(origin["valid"]["labels"].shape)
    print("labels:", torch.allclose(valid_feature_labels.cpu().unsqueeze(1).unsqueeze(2),origin["valid"]["labels"].to(torch.float32)))
    print("labels:", torch.allclose(test_feature_labels.cpu().unsqueeze(1).unsqueeze(2), origin["test"]["labels"].to(torch.float32)))
    print("labels:", torch.allclose(train_feature_labels.cpu().unsqueeze(1).unsqueeze(2), origin["train"]["labels"].to(torch.float32)))
    print(train_feature_labels)
    print(origin["train"]["labels"])
    if modality == 'physio':
        data_set = {
            "train": {
                "vision1": origin["train"]["vision1"],
                "vision2": origin["train"]["vision2"],
                "physio": train_feature_data.cpu().numpy(),
                "labels": train_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            },
            "valid": {
                "vision1": origin["valid"]["vision1"],
                "vision2": origin["valid"]["vision2"],
                "physio": valid_feature_data.cpu().numpy(),
                "labels": valid_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            },
            "test": {
                "vision1": origin["test"]["vision1"],
                "vision2": origin["test"]["vision2"],
                "physio": test_feature_data.cpu().numpy(),
                "labels": test_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            }
        }
    elif modality == 'vision1':
        data_set = {
            "train": {
                "vision1": train_feature_data.cpu().numpy(),
                "vision2": origin["train"]["vision2"],
                "physio": origin["train"]["text"],
                "labels": train_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            },
            "valid": {
                "vision1": valid_feature_data.cpu().numpy(),
                "vision2": origin["valid"]["vision2"],
                "physio": origin["valid"]["physio"],
                "labels": valid_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            },
            "test": {
                "vision1": test_feature_data.cpu().numpy(),
                "vision2": origin["test"]["vision2"],
                "physio": origin["test"]["physio"],
                "labels": test_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            }
        }
    elif modality == 'vision2':
        data_set = {
            "train": {
                "vision1": origin["train"]["vision1"],
                "vision2": train_feature_data.cpu().numpy(),
                "physio": origin["train"]["physio"],
                "labels": train_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            },
            "valid": {
                "vision1": origin["valid"]["vision1"],
                "vision2": valid_feature_data.cpu().numpy(),
                "physio": origin["valid"]["physio"],
                "labels": valid_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            },
            "test": {
                "vision1": origin["test"]["vision1"],
                "vision2": test_feature_data.cpu().numpy(),
                "physio": origin["test"]["physio"],
                "labels": test_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            }
        }
    else :
        print('wrong in feature generator!!!!')
        exit()
    print(data_set['train']['vision1'].shape)
    print(data_set['valid']['vision1'].shape)
    print(data_set['test']['vision1'].shape)
    
    print(data_set['train']['vision2'].shape)
    print(data_set['valid']['vision2'].shape)
    print(data_set['test']['vision2'].shape)
    
    print(data_set['train']['physio'].shape)
    print(data_set['valid']['physio'].shape)
    print(data_set['test']['physio'].shape)
    
    print(data_set['train']['labels'].shape)
    print(data_set['valid']['labels'].shape)
    print(data_set['test']['labels'].shape)
    index = dataset[len(dataset)-1]
    
    with open(f'data/{dataset}/{dataset[:-1]}_cl_{index}_{modality}.pkl', 'wb') as f:
        pickle.dump(data_set, f)