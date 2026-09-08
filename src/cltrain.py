import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from transformers import get_scheduler

from models.loss import NTXentLoss, SupConLoss
import tensorboard
from torch.utils.tensorboard import SummaryWriter

from dataloader.dataloader import data_generator

# ==================== 引入 TWMDA Loss 核心组件 ====================
def twmda_euclidean(x1, x2):
    return ((x1 - x2)**2).sum().sqrt()

def twmda_k_moment(output_s, output_t, k):
    output_s = (output_s**k).mean(0)
    output_t = (output_t**k).mean(0)
    return twmda_euclidean(output_s, output_t)

def twmda_alignment_loss(feat_s, feat_t, belta_moment=4):
    """TWMDA 高阶矩特征对齐"""
    s_mean = feat_s.mean(0)
    t_mean = feat_t.mean(0)
    feat_s = feat_s - s_mean
    feat_t = feat_t - t_mean

    reg_info = twmda_euclidean(feat_s, feat_t)
    for i in range(belta_moment - 1):
        reg_info = reg_info + twmda_k_moment(feat_s, feat_t, i + 2)
    return reg_info

def twmda_adentropy(logits, lamda=0.1):
    """TWMDA 对抗熵 (目标域熵最小化)"""
    out_t = F.softmax(logits, dim=1)
    # TWMDA 的熵计算逻辑
    loss_adent = lamda * torch.mean(torch.sum(out_t * (torch.log(out_t + 1e-5)), 1))
    return loss_adent
# ==================================================================


def Trainer(data_path, configs, dataset, model, temporal_contr_model, sample_projector, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode, modality, pairwise):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    #scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, mode='min', verbose=True)
    scheduler1 = get_scheduler(
      "cosine",
      optimizer=model_optimizer,
      num_warmup_steps=20,
      num_training_steps=1000
    )
    # scheduler2 = get_scheduler(
    #   "cosine",
    #   optimizer=temp_cont_optimizer,
    #   num_warmup_steps=20,
    #   num_training_steps=1000
    # )

    writer = SummaryWriter(log_dir = experiment_log_dir)

    if  'pairwise' in training_mode and pairwise == 0:
        feature_generator(dataset, model, train_dl, valid_dl, test_dl, device, training_mode, modality)
        sys.exit()


    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, sample_projector, model_optimizer, temp_cont_optimizer,
                                             epoch,
                                            criterion, train_dl, config, device, training_mode, writer)
        writer.add_scalar('Train Loss', train_loss,epoch)
        writer.add_scalar('Train Accuracy', train_acc,epoch)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        scheduler1.step()
        # scheduler2.step()
        writer.add_scalar('Learning Rate1', scheduler1.optimizer.param_groups[0]['lr'],epoch)
        # writer.add_scalar('Learning Rate2', scheduler2.optimizer.param_groups[0]['lr'],epoch)
        if training_mode != 'self_supervised' and training_mode != 'SupCon':
            scheduler1.step(valid_loss)
            writer.add_scalar('Learning Rate1', scheduler1.optimizer.param_groups[0]['lr'],epoch)

        writer.add_scalar('Valid Loss', valid_loss,epoch)
        writer.add_scalar('Valid Accuracy', valid_acc,epoch)
        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')


    # save the model after training ...
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if (training_mode != "self_supervised") and (training_mode != "SupCon"):
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        writer.add_scalar('Test Loss', test_loss,epoch)
        writer.add_scalar('Test Accuracy', test_acc,epoch)
        logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')
    writer.close()
    if pairwise == 1:
        train_dl, valid_dl, test_dl = data_generator(data_path, configs, 'self_supervised', dataset, modality)
        feature_generator(dataset, model, train_dl, valid_dl, test_dl, device, training_mode, modality)
    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, sample_projector, model_optimizer, temp_cont_optimizer,  epoch, criterion, train_loader, config,
                device, training_mode, writer):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()
    if sample_projector is not None:
        sample_projector.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        # print(data)
        # print(labels)
        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised" or training_mode == "SupCon":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)
            # keep raw encoder features (before normalize) for true-SimCLR sample loss
            features1_raw = features1
            features2_raw = features2

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(features2, features1)


        if training_mode == "self_supervised":
            # ===========================ORIGIN==============================================================
            lambda1 = 1
            lambda2 = 1
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss1 = temp_cont_loss1 + temp_cont_loss2

            # sample loss mode switch.
            sample_loss_mode = getattr(config, "sample_loss_mode", "default")
            if sample_loss_mode == "sample":
                z1 = F.normalize(sample_projector(features1_raw), dim=1)
                z2 = F.normalize(sample_projector(features2_raw), dim=1)
                loss2 = nt_xent_criterion(z1, z2)
            else:
                loss2 = nt_xent_criterion(temp_cont_feat1, temp_cont_feat2)
            loss = lambda1*loss1+lambda2*loss2
            # =================================================================================================


            # ===========================SimCLR==============================================================
            # # 1. 关键修复：把 3D 张量展平成 2D 的特征向量 (Batch, Channels * Length)
            # z1 = features1.reshape(features1.shape[0], -1)
            # z2 = features2.reshape(features2.shape[0], -1)

            # # 2. SimCLR 的标准操作：算对比损失前，一定要做一次 L2 归一化
            # z1 = F.normalize(z1, dim=1)
            # z2 = F.normalize(z2, dim=1)

            # # 3. 计算纯净的样本对比损失
            # nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
            #                                config.Context_Cont.use_cosine_similarity)
            # loss = nt_xent_criterion(z1, z2)
            # =================================================================================================

            # ===========================SimSiam==============================================================
            # # 1. 展平特征 (把 3D 时序张量拉平成 2D 向量)
            # z1 = features1.reshape(features1.shape[0], -1)
            # z2 = features2.reshape(features2.shape[0], -1)

            # # 2. 动态构建 Predictor (仅在第一轮的第一个 Batch 触发一次)
            # # 为了不修改 model.py，我们把 predictor 动态挂载到 model 上
            # if not hasattr(model, 'simsiam_predictor'):
            #     in_dim = z1.shape[1]
            #     # 经典的 SimSiam 预测头结构: Linear -> BN -> ReLU -> Linear
            #     # 隐藏层维度通常设置为输入维度的 1/4
            #     model.simsiam_predictor = nn.Sequential(
            #         nn.Linear(in_dim, in_dim // 4, bias=False),
            #         nn.BatchNorm1d(in_dim // 4),
            #         nn.ReLU(inplace=True),
            #         nn.Linear(in_dim // 4, in_dim)
            #     ).to(device)

            #     # 必须把新生 predictor 的参数加入到原来的优化器中，否则它不更新
            #     model_optimizer.add_param_group({'params': model.simsiam_predictor.parameters()})

            # # 3. 前向传播：计算预测值
            # p1 = model.simsiam_predictor(z1)
            # p2 = model.simsiam_predictor(z2)

            # # 4. 定义 SimSiam 核心损失函数 (负余弦相似度)
            # def simsiam_loss(p, z):
            #     # 【极其关键】：z 必须切断梯度 (Stop-gradient)
            #     z = z.detach()
            #     # 余弦相似度计算前需要 L2 归一化
            #     p = F.normalize(p, dim=1)
            #     z = F.normalize(z, dim=1)
            #     # 负的点乘之和就是负的余弦相似度
            #     return -(p * z).sum(dim=1).mean()

            # # 5. 计算对称的极小化损失
            # # 左脑预测右脑，右脑预测左脑
            # loss = 0.5 * simsiam_loss(p1, z2) + 0.5 * simsiam_loss(p2, z1)
            # =================================================================================================



            # # ============================TWMDA============================================================
            # # 1. 计算特征的高阶矩匹配损失 (多源域对齐的核心逻辑)
            # # 使用经过 temporal_contr_model 提取后的时序特征进行对齐
            # loss_moment = twmda_alignment_loss(temp_cont_feat1, temp_cont_feat2, belta_moment=4)

            # # 2. 计算目标视图的对抗熵 (熵最小化)
            # # 注意: 这里用 predictions2，因为 TWMDA 需要对分类器的输出做熵约束
            # loss_entropy = twmda_adentropy(predictions2, lamda=0.1)

            # # 3. 组合 Loss (0.5 为 TWMDA 原文设定的矩匹配权重)
            # loss = 0.5 * loss_moment + loss_entropy
            # # =================================================================================================

        elif training_mode == "SupCon":
            lambda1 = 0.01
            lambda2 = 0.1
            Sup_contrastive_criterion = SupConLoss(device)

            supCon_features = torch.cat([temp_cont_feat1.unsqueeze(1), temp_cont_feat2.unsqueeze(1)], dim=1)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + Sup_contrastive_criterion(supCon_features,
                                                                                             labels) * lambda2

        else:
            output = model(data)
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
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

            if (training_mode == "self_supervised") or (training_mode == "SupCon"):
                pass
            else:
                output = model(data)

            # compute loss
            if (training_mode != "self_supervised") and (training_mode != "SupCon"):
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_loss = 0
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        return total_loss, total_acc, outs, trgs




def gen_pseudo_labels(model, dataloader, device, experiment_log_dir, modality):
    from sklearn.metrics import accuracy_score
    model.eval()
    softmax = nn.Softmax(dim=1)

    # saving output data
    all_pseudo_labels = np.array([])
    all_labels = np.array([])
    all_data = []

    with torch.no_grad():
        for data, labels, _, _ in dataloader:
            data = data.float().to(device)
            labels = labels.view((-1)).long().to(device)

            # forward pass
            predictions, features = model(data)

            normalized_preds = softmax(predictions)
            pseudo_labels = normalized_preds.max(1, keepdim=True)[1].squeeze()
            all_pseudo_labels = np.append(all_pseudo_labels, pseudo_labels.cpu().numpy())

            all_labels = np.append(all_labels, labels.cpu().numpy())
            all_data.append(data)

    all_data = torch.cat(all_data, dim=0)

    data_save = dict()
    data_save["samples"] = all_data
    data_save["labels"] = torch.LongTensor(torch.from_numpy(all_pseudo_labels).long())
    file_name = f"{modality}pseudo_train_data.pt"
    torch.save(data_save, os.path.join(experiment_log_dir, file_name))
    print("Pseudo labels generated ...")

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
    # print("labels:", torch.allclose(valid_feature_labels.cpu().unsqueeze(1).unsqueeze(2),origin["valid"]["labels"].to(torch.float32)))
    # print("labels:", torch.allclose(test_feature_labels.cpu().unsqueeze(1).unsqueeze(2), origin["test"]["labels"].to(torch.float32)))
    # print("labels:", torch.allclose(train_feature_labels.cpu().unsqueeze(1).unsqueeze(2), origin["train"]["labels"].to(torch.float32)))
    print(train_feature_labels)
    print(origin["train"]["labels"])
    if modality == 'physio':
        data_set = {
            "train": {
                "vision1": origin["train"]["vision1"],
                "vision2": origin["train"]["vision2"],
                "physio": train_feature_data.cpu().detach().numpy(),
                "labels": train_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            },
            "valid": {
                "vision1": origin["valid"]["vision1"],
                "vision2": origin["valid"]["vision2"],
                "physio": valid_feature_data.cpu().detach().numpy(),
                "labels": valid_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            },
            "test": {
                "vision1": origin["test"]["vision1"],
                "vision2": origin["test"]["vision2"],
                "physio": test_feature_data.cpu().detach().numpy(),
                "labels": test_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            }
        }
    elif modality == 'vision2':
        data_set = {
            "train": {
                "vision1": origin["train"]["vision1"],
                "vision2": train_feature_data.cpu().detach().numpy(),
                "physio": origin["train"]["physio"],
                "labels": train_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            },
            "valid": {
                "vision1": origin["valid"]["vision1"],
                "vision2": valid_feature_data.cpu().detach().numpy(),
                "physio": origin["valid"]["physio"],
                "labels": valid_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            },
            "test": {
                "vision1": origin["test"]["vision1"],
                "vision2": test_feature_data.cpu().detach().numpy(),
                "physio": origin["test"]["physio"],
                "labels": test_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            }
        }
    elif modality == 'vision1':
        data_set = {
            "train": {
                "vision1": train_feature_data.cpu().detach().numpy(),
                "vision2": origin["train"]["vision2"],
                "physio": origin["train"]["physio"],
                "labels": train_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            },
            "valid": {
                "vision1": valid_feature_data.cpu().detach().numpy(),
                "vision2": origin["valid"]["vision2"],
                "physio": origin["valid"]["physio"],
                "labels": valid_feature_labels.unsqueeze(1).unsqueeze(2).cpu().numpy()
            },
            "test": {
                "vision1": test_feature_data.cpu().detach().numpy(),
                "vision2": origin["test"]["vision2"],
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
    if 'SupCon' in training_mode:
        with open(f'data/{dataset}/{dataset[:-1]}_{modality}_supcon{index}.pkl', 'wb') as f:
            pickle.dump(data_set, f)
    elif 'self' in training_mode:
        with open(f'data/{dataset}/{dataset[:-1]}_{modality}_self{index}.pkl', 'wb') as f:
            pickle.dump(data_set, f)
    else:
        with open(f'data/{dataset}/{dataset[:-1]}_{modality}_ft{index}.pkl', 'wb') as f:
            pickle.dump(data_set, f)
