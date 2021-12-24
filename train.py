import time
import os

import numpy as np
import torch
import torch.nn as nn

from data.build_dataloader import build_dataloader
from scheduler.build_scheduler import build_scheduler
from utils.metrics import compute_metric
from models.vgg import VGG
from models.attention import SingleLayerLSTM
from config import Config

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    save_dir = './checkpoint'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config()

    ############### Please modify the model here ##############
    model = SingleLayerLSTM(**config.model_cfg).to(device)
    # model.load_state_dict(torch.load(os.path.join(save_dir, 'checkpoint_epoch_15.pth')))

    print("{:.2f}M parameters!".format(sum([np.prod(x.shape) for x in model.parameters()]) / 1000000))

    ############# You might want to use your own dataloader here ##############
    train_dataloader, valid_dataloader = build_dataloader(config.data_cfg)
    
    ############## Adjust the loss function and optimizer here ################
    train_loss_fn = nn.MSELoss().to(device)
    valid_loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=4e-3, weight_decay=0.01)
    lr_scheduler = build_scheduler(config.scheduler_cfg, optimizer)

    
    num_epoch = 100
    for epoch in range(num_epoch):
        train_one_epoch(epoch, model, train_dataloader, optimizer, train_loss_fn, lr_scheduler=lr_scheduler, log_interval=1)
        validate(model, valid_dataloader, valid_loss_fn)

        if lr_scheduler is not None:
            lr_scheduler.epoch_update(epoch + 1)

        torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint_epoch_{}.pth'.format(epoch + 1)))


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, lr_scheduler=None, log_interval=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (X, y_arousal, y_valence) in enumerate(loader):
        last_batch = batch_idx == last_idx
        
        pred_arousal, pred_valence = model(X.to(device))
        loss = loss_fn(pred_arousal, y_arousal) + loss_fn(pred_valence, y_valence)

        losses_m.update(loss.item(), pred_arousal.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']

            print(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:#.4g} ({loss.avg:#.3g}) '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s    '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)    '
                'LR: {lr:.3e} '.format(
                    epoch,
                    batch_idx + 1,
                    len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=pred_arousal.size(0) / batch_time_m.val,
                    rate_avg=pred_arousal.size(0) / batch_time_m.avg,
                    lr=lr
                )
            )

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates)

        end = time.time()


def validate(model, loader, loss_fn):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        y_arousal_all = []
        y_valence_all = []
        pred_arousal_all = []
        pred_valence_all = []

        for batch_idx, (X, y_arousal, y_valence) in enumerate(loader):
            pred_arousal, pred_valence = model(X.to(device))

            y_arousal_all.append(y_arousal.numpy())
            y_valence_all.append(y_valence.numpy())
            pred_arousal_all.append(pred_arousal.detach().numpy())
            pred_valence_all.append(pred_valence.detach().numpy())

            loss = loss_fn(pred_arousal, y_arousal.to(device)) + loss_fn(pred_valence, y_valence.to(device))
            losses_m.update(loss.item(), pred_arousal.size(0))

            batch_time_m.update(time.time() - end)

            end = time.time()
        
        y_arousal = np.hstack(y_arousal_all)
        y_valence = np.hstack(y_valence_all)
        pred_arousal = np.hstack(pred_arousal_all)
        pred_valence = np.hstack(pred_valence_all)

    output = dict(arousal=pred_arousal, valence=pred_valence)
    target = dict(arousal=y_arousal, valence=y_valence)
    results = compute_metric(output, target)

    print(
        'Valid  '
        'Loss: {loss.avg:#.3g} '
        'Time: {batch_time.avg:.3f}s   '
        'RMSE(A): {:.4f}   '
        'RMSE(V): {:.4f}   '
        'R2(A): {:.4f}   '
        'R2(V): {:.4f}   '.format(
            results['rmse_arousal'],
            results['rmse_valence'],
            results['r2_arousal'],
            results['r2_valence'],
            loss=losses_m,
            batch_time=batch_time_m
            )
        )



if __name__ == '__main__':
    main()