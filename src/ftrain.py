import torch
from torch import nn
import sys
from models import fusion
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from src.confusion_matrix import make_confusion_matrix
####################################################################
#
# Construct the model and the CTC module (which may not be needed)
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = fusion.fusionModel(hyp_params)
    
    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.6, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    scheduler = settings['scheduler']
    
    writer = SummaryWriter(log_dir=f'logs/{hyp_params.name}{hyp_params.mode}{hyp_params.index}')

    # text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
    dataset = hyp_params.dataset
    def train(model, optimizer, criterion):
        if hyp_params.check:
            model = load_model(hyp_params, name=hyp_params.name+str(hyp_params.num_epochs)+hyp_params.mode)
            _, results, truths = evaluate(model, criterion, test=True)
            eval_private(results, truths)
            make_confusion_matrix(truths, results, hyp_params.class_name, dataset, hyp_params.index, hyp_params.mode)
            return 
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, physio, vision2, vision1 = batch_X
            # print("train function:")
            # print(f'sample_index: {i_batch}')
            # print(f'p: {physio.shape}')
            # print(f'v2: {vision2.shape}')
            # print(f'v1: {vision1.shape}')
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1

            model.zero_grad()
                
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    physio, vision1, vision2, eval_attr = physio.cuda(), vision1.cuda(), vision2.cuda(), eval_attr.cuda()
                    
                    eval_attr = eval_attr.long() 
            
            batch_size = physio.size(0)            
                
            combined_loss = 0
            net = model

            # cross attention
            if torch.isnan(physio).any():
                print("wrong")
            preds, hiddens = net(physio, vision2, vision1)
            
            preds = preds.view(-1,hyp_params.output_dim)
            eval_attr = eval_attr.view(-1)
            
            raw_loss = criterion(preds, eval_attr)
            combined_loss = raw_loss
            raw_loss.backward()
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
    
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):# 在这里是一个个batch
                sample_ind, physio, vision2, vision1 = batch_X
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
                
                eval_attr = eval_attr.long()
                
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        physio, vision1, vision2, eval_attr = physio.cuda(), vision1.cuda(), vision2.cuda(), eval_attr.cuda()
                        
                batch_size = physio.size(0)
                
                net = model
                preds, _ = net(physio, vision2, vision1)
                

                preds = preds.view(-1,hyp_params.output_dim)
                eval_attr = eval_attr.view(-1)
                
                total_loss += criterion(preds, eval_attr).item() * batch_size
                # print("eval_attr")
                # print(eval_attr)
                # Collect the results into dictionary

                results.append(preds)
                truths.append(eval_attr)
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths
    active_time = datetime.now()
    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_loss = train(model, optimizer, criterion)
        if hyp_params.check:
            return
        val_loss, val_results, val_truths = evaluate(model, criterion, test=False)
        test_loss, test_results, test_truths = evaluate(model, criterion, test=True)
        writer.add_scalar('Train Loss', train_loss, epoch)
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss
        writer.add_scalar('Learning Rate', scheduler.optimizer.param_groups[0]['lr'],epoch)
        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*50)
        
        print("Valid Results")
        valid_acc, _ = eval_private(val_results, val_truths)
        print("Test Results")
        test_acc, _= eval_private(test_results, test_truths)
        sys.stdout.flush()
        writer.add_scalar('Valid Accuracy', valid_acc, epoch)
        writer.add_scalar('Test Accuracy', test_acc, epoch)
        if epoch == hyp_params.num_epochs:
            make_confusion_matrix(test_truths, test_results, hyp_params.class_name, dataset, hyp_params.index, hyp_params.mode)
        
        if val_loss < best_valid:
            print(f"PRIVATE TRAINED MODEL SAVED IN {hyp_params.name}{hyp_params.num_epochs}{hyp_params.mode}.pt!")
            save_model(hyp_params, model, name=hyp_params.name+str(hyp_params.num_epochs)+hyp_params.mode)
            best_valid = val_loss
    model = load_model(hyp_params, name=hyp_params.name+str(hyp_params.num_epochs)+hyp_params.mode)
    _, results, truths = evaluate(model, criterion, test=True)
    # print(f"result:{results.shape}")
    # print(f"truths:{truths.shape}")
    
    print("FINAL RESULTS")
    eval_private(results, truths)
    if hyp_params.num_epochs == 1:
        make_confusion_matrix(truths, results, hyp_params.class_name, dataset, hyp_params.index, hyp_params.mode)
    
    sys.stdout.flush()
    end_time = datetime.now()
    print("  -START TIME")
    print(active_time)
    print("  -END TIME")
    print(end_time)
    writer.close()