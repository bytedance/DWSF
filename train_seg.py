# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Inc. and/or its affiliates
# ------------------------------------------------------------------------
import argparse
import os
from utils.dataset import *
from utils.util import setup_seed
from networks.segmentation.model import U2NETP
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Training settings
parser = argparse.ArgumentParser(description='option')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--H', type=int, default=512)
parser.add_argument('--W', type=int, default=512)
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('-optimtype', type=str, default='Adamw', help='SGD,Adam,Adamx,Adamw')
parser.add_argument('-lr_schedulertype', type=str, default='CosineAnnealingWarmRestarts', help='CosineAnnealingWarmRestarts,MultiStepLR,ReduceLROnPlateau,CyclicLR')
parser.add_argument('--train_path', type=str, default='./')
parser.add_argument('--val_path', type=str, default='./')
parser.add_argument('--output_path', type=str, default='./')


bce_loss = torch.nn.BCEWithLogitsLoss(reduce=False)
def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v) * (labels_v + 1)
    loss1 = bce_loss(d1, labels_v) * (labels_v + 1)
    loss2 = bce_loss(d2, labels_v) * (labels_v + 1)
    loss3 = bce_loss(d3, labels_v) * (labels_v + 1)
    loss4 = bce_loss(d4, labels_v) * (labels_v + 1)
    loss5 = bce_loss(d5, labels_v) * (labels_v + 1)
    loss6 = bce_loss(d6, labels_v) * (labels_v + 1)
    loss = torch.mean(loss0 + 0.25 * (loss1 + loss2 + loss3 + loss4 + loss5 + loss6))
    return loss


def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


def muti_iou_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    iou0 = iou_loss(d0, labels_v)
    iou1 = iou_loss(d1, labels_v)
    iou2 = iou_loss(d2, labels_v)
    iou3 = iou_loss(d3, labels_v)
    iou4 = iou_loss(d4, labels_v)
    iou5 = iou_loss(d5, labels_v)
    iou6 = iou_loss(d6, labels_v)
    loss = torch.mean(iou0 + 0.25*(iou1 + iou2 + iou3 + iou4 + iou5 + iou6))
    return loss


def train_epoch(train_loader, model, optimizer, epoch):
    model.train()
    train_bce_loss = 0.
    train_iou_loss = 0.
    for i, (image, target, index, _) in enumerate((train_loader)):
        image, target = image.cuda(), target.cuda()
        d0, d1, d2, d3, d4, d5, d6 = model(image)
        loss_bce = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, target)
        loss_iou = muti_iou_loss_fusion(d0, d1, d2, d3, d4, d5, d6, target) * 0.1
        loss = loss_bce + loss_iou

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_bce_loss += loss_bce.item()
        train_iou_loss += loss_iou.item()

    print("[Train]Epoch: {}, BCE loss: {}, ".format(epoch, train_bce_loss))
    print("[Train]Epoch: {}, IOU loss: {}, ".format(epoch, train_iou_loss))
    return None


def testcheckpoint(test_loader, model):
    model.eval()
    val_bce_loss = 0.
    val_iou_loss = 0.
    with torch.no_grad():
        for data, target,index,_ in test_loader:
            data, target = data.cuda(), target.cuda()
            d0, d1, d2, d3, d4, d5, d6 = model(data)
            loss_bce = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, target)
            loss_iou = muti_iou_loss_fusion(d0, d1, d2, d3, d4, d5, d6, target) * 0.1
            val_bce_loss += loss_bce.item()
            val_iou_loss += loss_iou.item()
    print("[Val]Epoch: {}, BCE loss: {}, ".format(val_bce_loss))
    print("[Val]Epoch: {}, IOU loss: {}, ".format(val_iou_loss))

    return None

def save_checkpoint(state, epoch, file_name):
    torch.save(state, file_name)


def main(args):
    # build model
    model = U2NETP(mode='train').cuda()

    # select optimizer
    if args.optimtype == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimtype == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimtype == 'Adamx':
        optimizer = torch.optim.Adamax(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimtype == 'Adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # select scheduler
    if args.lr_schedulertype == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)
    elif args.lr_schedulertype == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    elif args.lr_schedulertype == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif args.lr_schedulertype == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=(1 / 4) * args.lr, max_lr=args.lr, cycle_momentum=True)

    train_set = SegDataset(path_wm=args.train_path+'/img/', path_mask=args.train_path+'/mask/', tmp_path='tmp1')
    val_set = SegDataset(path_wm=args.val_path+'/img/', path_mask=args.val_path+'/mask/', tmp_path='tmp1')
    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        train_epoch(train_loader, model, optimizer, epoch)
        scheduler.step()

        testcheckpoint(val_loader, model)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, epoch, args.output_path+'/seg_' + str(epoch) + '.pth')


if __name__ == '__main__':
    setup_seed(20)
    args = parser.parse_args()
    main(args)

