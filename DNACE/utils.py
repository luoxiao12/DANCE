"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import os.path as osp
import time
from PIL import Image
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

sys.path.append('../../..')
import tllib.vision.datasets as datasets
import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import open_accuracy,accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.datasets.imagelist import MultipleDomainsDataset
from cluster import PairEnum
import math
from plabel_allocator import *

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
def curriculum_scheduler(t, T, begin=0, end=1, mode='linear', func=None):
    """
    ratio \in [0,1]
    """
    pho = t/T
    if mode == 'linear':
        ratio = pho
    elif mode == 'exp':
        # ratio = 1 - math.exp(-5*pho)
        ratio = 1 - math.exp(-4*pho)
    elif mode == 'customize':
        ratio = func(t, T)
    budget = begin + ratio * (end-begin)
    return budget, pho




def CSOT_PL(net, eval_loader, num_class, batch_size, feat_dim=512, budget=1., sup_label=None, 
              reg_feat=1, reg_lab=1, version='fast', Pmode='logP', reg_e=0.1, reg_sparsity=None):
    net.eval()

    all_pseudo_labels = torch.zeros((len(eval_loader.dataset), num_class), dtype=torch.float64).cuda()
    all_gt_labels = torch.zeros((len(eval_loader.dataset),), dtype=torch.int64).cuda()
    all_selected_mask = torch.zeros((len(eval_loader.dataset)), dtype=torch.bool).cuda()
    all_conf = torch.zeros((len(eval_loader.dataset),), dtype=torch.float64).cuda()
    all_argmax_plabels = torch.zeros((len(eval_loader.dataset),), dtype=torch.int64).cuda()
    # loading given samples
    for batch_idx, (inputs, labels,  index) in enumerate(eval_loader):
        feat, out, _, _ = net(inputs.cuda())
        index = index.cuda()

        if sup_label is not None:
            L = torch.eye(num_class, dtype=torch.float64)[sup_label[index]].cuda()
        else:
            L = torch.eye(num_class, dtype=torch.float64)[labels].cuda()

        if Pmode == 'out':
            P = out
        if Pmode == 'logP':
            P = F.log_softmax(out, dim=1)
        if Pmode == 'softmax':
            P = F.softmax(out, dim=1)
        
        norm_feat = F.normalize(feat)
        couplings, selected_mask = curriculum_structure_aware_PL(norm_feat.detach(), P.detach(), top_percent=budget, L=L, 
                                                                reg_feat=reg_feat, reg_lab=reg_lab, version=version, reg_e=reg_e,
                                                                reg_sparsity=reg_sparsity)
        
        all_gt_labels[index] = labels.cuda()
        all_selected_mask[index] = selected_mask

        row_sum = torch.sum(couplings, 1).reshape((-1,1))
        pseudo_labels = torch.div(couplings, row_sum)
        max_value, argmax_plabels = torch.max(couplings, axis=1)
        conf = max_value / (1/couplings.size(0))
        conf = torch.clip(conf, min=0, max=1.0)

        all_conf[index] = conf
        all_pseudo_labels[index, :] = pseudo_labels
        all_argmax_plabels[index] = argmax_plabels

    return all_pseudo_labels, all_argmax_plabels,  all_conf,all_gt_labels



sim_list = []
def get_ulb_sim_matrix(mode, sim_matrix_ulb, cluster_preds_t, update_list=True):
    if mode == 'stats':
        return sim_matrix_ulb, 0, 0
    elif mode == 'argmax':
        y_c_t = cluster_preds_t.argmax(dim=1).contiguous().view(-1, 1)
        sim_matrix_ulb_full = torch.eq(y_c_t, y_c_t.T).float().to(cluster_preds_t.device)
        sim_matrix_ulb_full = (sim_matrix_ulb_full - 0.5) * 2
        sim_matrix_ulb_full = sim_matrix_ulb_full.flatten()
        return sim_matrix_ulb_full
    else:
        if mode == 'sim':
            feat_row, feat_col = PairEnum(F.normalize(cluster_preds_t, dim=1))
        elif mode == 'prob':
            feat_row, feat_col = PairEnum(F.softmax(cluster_preds_t, dim=1))
        tmp_distance_ori = torch.bmm(
            feat_row.view(feat_row.size(0), 1, -1),
            feat_col.view(feat_row.size(0), -1, 1)
        )
        sim_threshold = 0.92
        sim_ratio = 0.5 / 12
        diff_ratio = 5.5 / 12
        similarity = tmp_distance_ori.squeeze()
        if update_list:
            global sim_list
            sim_list.append(similarity)
            if len(sim_list) > 30:
                sim_list = sim_list[1:]
        sim_all = torch.cat(sim_list, dim=0)
        sim_all_sorted, _ = torch.sort(sim_all)

        n_diff = min(len(sim_all) * diff_ratio, len(sim_all)-1)
        n_sim = min(len(sim_all) * sim_ratio, len(sim_all))

        low_threshold = sim_all_sorted[int(n_diff)]
        high_threshold = max(sim_threshold, sim_all_sorted[-int(n_sim)])

        sim_matrix_ulb = torch.zeros_like(similarity).float()

        if high_threshold != low_threshold:
            sim_matrix_ulb[similarity >= high_threshold] = 1.0
            sim_matrix_ulb[similarity <= low_threshold] = -1.0
        else:
            sim_matrix_ulb[similarity > high_threshold] = 1.0
            sim_matrix_ulb[similarity < low_threshold] = -1.0
        return sim_matrix_ulb, low_threshold, high_threshold
    
def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def shrink_pred(pseudo_label,  conf_thresh = 0.9):
    
    B, C = pseudo_label.shape
    #print('unknown_pred:', unknown_pred.shape)
    _, pred = pseudo_label.topk(1, 1, True, True)
    
    max_probs = pseudo_label.max(dim=-1)[0]
    mask = pseudo_label.ge(conf_thresh).float()

    sorted_prob_w, sorted_idx = pseudo_label.topk(C, dim=-1, sorted=True)
    #print(sorted_prob_w)
    # organize logit_s same as the sorted_prob_w
    
    if mask.mean().item() == 1: # no uncertain samples to shrink
        pred = pred
    else:
        for b in range(B):
            if max_probs[b] >= conf_thresh: # skip certain samples
                continue
            # iteratively remove classes to enhance confidence until satisfying the confidence threshold
            c = int(C/2)
            # new confidence in the shrunk class space (classes ranging from 1 ~ (c-1) are removed)
            sub_conf = sorted_prob_w[b, 0] / (sorted_prob_w[b, 0] + sorted_prob_w[b, c:].sum())
            #print('sub_conf:',sub_conf)
            
            # break either when satifying the threshold or traversing to the final class (with smallest value)
            if (sub_conf <= 0.4) :
                pred[b] = -1

    
    return pred

def validate(val_loader, model, args, device) -> float:
    
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            _,output,_,_ = model(images)
            
            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg


def open_validate(val_loader, model, args, device) -> float:
    novel_samples = []
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            _,output,_,_ = model(images)   
            output = F.softmax(output,dim=-1)
            pred = shrink_pred(output, 0.9)
            
            # measure accuracy and record loss
            acc1, = open_accuracy(pred, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg, novel_samples




def empirical_risk_minimization(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def confidence_update_lw(model, confidence, batchX, batchY, batch_index):
    with torch.no_grad():
        device = batchX.device
        _,batch_outputs,_,_ = model(batchX)
        sm_outputs = F.softmax(batch_outputs, dim=1)

        onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
        onezero[batchY > 0] = 1
        counter_onezero = 1 - onezero
        onezero = onezero.to(device)
        counter_onezero = counter_onezero.to(device)

        new_weight1 = sm_outputs * onezero
        new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(
            confidence.shape[1], 1).transpose(0, 1)
        new_weight2 = sm_outputs * counter_onezero
        new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(
            confidence.shape[1], 1).transpose(0, 1)
        new_weight = new_weight1 + new_weight2

        confidence[batch_index, :] = new_weight
        return confidence
    
def mask_generate(num_classes, max_probs, max_idx, batch, threshold):
    mask_ori = torch.zeros(batch).cuda()
    for i in range(num_classes):
        idx = np.where(max_idx.cpu() == i)[0]
        #print(idx)
        m = max_probs[idx].ge(threshold[i]).float()
        for k in range(len(idx)):
            mask_ori[idx[k]]+=m[k]
    return mask_ori.cuda()

def EMA(cur_weight, past_weight, momentum=0.9):
    new_weight = momentum * past_weight + (1 - momentum) * cur_weight
    return new_weight

def custom_cross_entropy(input, target):
    # 计算交叉熵损失
    loss = F.cross_entropy(input, target, reduction='none')

    # 计算每个类别的损失
    num_classes = input.size(1)
    class_losses = torch.zeros(num_classes, dtype=loss.dtype, device=loss.device) + 1e-8

    for c in range(num_classes):
        class_mask = (target == c).float()
        class_losses[c] = torch.sum(loss * class_mask)

    return class_losses

def lws_loss(outputs, partialY, confidence, index, lw_weight, lw_weight0, epoch_ratio):
    device = outputs.device

    onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
    onezero[partialY > 0] = 1
    counter_onezero = 1 - onezero
    onezero = onezero.to(device)
    counter_onezero = counter_onezero.to(device)

    sig_loss1 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss1 = sig_loss1.to(device)
    sig_loss1[outputs < 0] = 1 / (1 + torch.exp(outputs[outputs < 0]))
    sig_loss1[outputs > 0] = torch.exp(-outputs[outputs > 0]) / (
        1 + torch.exp(-outputs[outputs > 0]))
    l1 = confidence[index, :] * onezero * sig_loss1
    average_loss1 = torch.sum(l1) / l1.size(0)

    sig_loss2 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss2 = sig_loss2.to(device)
    sig_loss2[outputs > 0] = 1 / (1 + torch.exp(-outputs[outputs > 0]))
    sig_loss2[outputs < 0] = torch.exp(
        outputs[outputs < 0]) / (1 + torch.exp(outputs[outputs < 0]))
    l2 = confidence[index, :] * counter_onezero * sig_loss2
    average_loss2 = torch.sum(l2) / l2.size(0)

    average_loss = lw_weight0 * average_loss1 + lw_weight * average_loss2
    return average_loss, lw_weight0 * average_loss1, lw_weight * average_loss2

class Difficulty:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(self.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        return weights

    def custom_cross_entropy(self,input, target):
        loss = F.cross_entropy(input, target, reduction='none')

        num_classes = input.size(1)
        class_losses = torch.zeros(num_classes, dtype=loss.dtype, device=loss.device)

        for c in range(num_classes):
            class_mask = (target == c).float()
            class_losses[c] = torch.sum(loss * class_mask+1e-8)

        return class_losses
    
    def cal_weights(self, pred,  label):
        cur_dice = self.custom_cross_entropy(pred, label)
        delta_dice = cur_dice - self.last_dice
        cur_cls_learn = torch.where(delta_dice>0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        cur_cls_unlearn = torch.where(delta_dice<=0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)

        self.last_dice = cur_dice

        self.cls_learn = EMA(cur_cls_learn, self.cls_learn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)

        cur_diff = torch.pow(cur_diff, 1/5)

        self.dice_weight = EMA(1. - cur_dice, self.dice_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        weights = cur_diff * self.dice_weight
        return weights * self.num_cls