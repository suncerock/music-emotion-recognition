import time
import os

import numpy as np
import torch
import torch.nn as nn

from utils.opensmile_dataloader import build_opensmile_dynamic_dataloaer
from utils.metrics import compute_metric  # TODO: support kendall's tau, fix R^2, !! only use RMSE now !!
from models.linear import Linear


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

    ############### Please modify the model here ##############
    model = Linear().to(device)
    # model.load_state_dict(torch.load(os.path.join(save_dir, 'checkpoint_epoch_15.pth')))

    print("{:.2f}K parameters!".format(sum([np.prod(x.shape) for x in model.parameters()]) / 1000))

    ############# You might want to use your own dataloader here ##############
    train_dataloader, valid_dataloader = build_opensmile_dynamic_dataloaer(batch_size=64)
    
    ############## Adjust the loss function and optimizer here ################
    train_loss_fn = nn.MSELoss().to(device)
    valid_loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    
    num_epoch = 50
    for epoch in range(num_epoch):
        train_one_epoch(epoch, model, train_dataloader, optimizer, train_loss_fn, log_interval=100)
        validate(model, valid_dataloader, valid_loss_fn)
        torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoint_epoch_{}.pth'.format(epoch + 1)))

        

def train_one_epoch(epoch, model, loader, optimizer, loss_fn, lr_scheduler=None, log_interval=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (X, y_arousal, y_valence, music_id) in enumerate(loader):
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
        music_id_all = []
        for batch_idx, (X, y_arousal, y_valence, music_id) in enumerate(loader):
            pred_arousal, pred_valence = model(X.to(device))


            y_arousal_all.append(y_arousal.numpy())
            y_valence_all.append(y_valence.numpy())
            pred_arousal_all.append(pred_arousal.detach().numpy())
            pred_valence_all.append(pred_valence.detach().numpy())
            music_id_all.append(music_id)

            loss = loss_fn(pred_arousal, y_arousal.to(device)) + loss_fn(pred_valence, y_valence.to(device))
            losses_m.update(loss.item(), pred_arousal.size(0))

            batch_time_m.update(time.time() - end)

            end = time.time()
        
        y_arousal = np.hstack(y_arousal_all)
        y_valence = np.hstack(y_valence_all)
        pred_arousal = np.hstack(pred_arousal_all)
        pred_valence = np.hstack(pred_valence_all)
        music_id = np.hstack(music_id_all)
    
    output = dict(arousal=pred_arousal, valence=pred_valence)
    target = dict(arousal=y_arousal, valence=y_valence)
    results = compute_metric(output, target, music_id)

    print(
        'Valid  '
        'Loss: {loss.avg:#.3g} '
        'Time: {batch_time.avg:.3f}s,'.format(
            loss=losses_m,
            batch_time=batch_time_m
            )
        )
    print(
        """
        Result\tRMSE by seg\tR2 by seg\tRMSE by song\tR2 by song
        Arousal\t{:11.2f}\t{:9.2f}\t{:12.2f}\t{:10.2f}
        Valence\t{:11.2f}\t{:9.2f}\t{:12.2f}\t{:10.2f}
        """.format(
            results['rmse_segments_arousal'], results['r2_segments_arousal'], results['rmse_songs_arousal'], results['r2_songs_arousal'],
            results['rmse_segments_valence'], results['r2_segments_valence'], results['rmse_songs_valence'], results['r2_songs_valence']
        )
    )



if __name__ == '__main__':
    main()