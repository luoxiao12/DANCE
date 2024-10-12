import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import random
import time
import warnings
import argparse
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import numpy as np
import utils
import shutil
from tllib.self_training.pseudo_label import ConfidenceBasedSelfTrainingLoss
from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature_ours, tsne, a_distance
from scdataset import Prepare_scDataloader
from scmodel import sc_net
from utils import  curriculum_scheduler, CSOT_PL
from entropy import TsallisEntropy
from copy import deepcopy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

def proden_loss(output1, target, true, eps=1e-12):
    output = F.softmax(output1, dim=1)
    l = target * torch.log(output)
    loss = (-torch.sum(l)) / l.size(0)

    revisedY = target.clone()
    revisedY[revisedY > 0] = 1
    # revisedY = revisedY * (output.clone().detach())
    revisedY = revisedY * output
    revisedY = revisedY / (revisedY).sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)
    new_target = revisedY

    return loss, new_target

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        #torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    labeled_train_source_loader,unlabeled_train_source_loader, test_source_loader, train_target_loader, test_target_loader, gene_size, type_num = Prepare_scDataloader(args).getloader()

    
    labeled_train_source_iter = ForeverDataIterator(labeled_train_source_loader)
    unlabeled_train_source_iter = ForeverDataIterator(unlabeled_train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    
    # encoder = sc_encoder(gene_size).cuda()
    # classifier = sc_classifier(type_num).cuda()
    classifier = sc_net(gene_size,type_num).cuda()
    
    # define optimizer and lr scheduler
    optimizer = SGD(classifier.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                    nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier).to(device)
        args.label_ratio = 1
        labeled_train_source_loader,_, _, _, _, _, _ = Prepare_scDataloader(args).getloader()
        source_feature = collect_feature_ours(labeled_train_source_loader, feature_extractor, device)
        target_feature = collect_feature_ours(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1, novel_samples = utils.validate(test_target_loader, classifier, args, device)
        print(acc1)
        return

    
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr())
        # train for one epoch
        if epoch < args.warm_up:
            print(f"warm up epoch: {epoch}/{args.warm_up}")
            warmup(labeled_train_source_iter, classifier, optimizer, lr_scheduler, epoch,type_num, args)
            
        
        else:
            
            
            budget, pho = curriculum_scheduler(epoch-args.warm_up, args.epochs-args.warm_up, 
                                    begin=0.3, end=1, mode='linear')
            print(f"current budget = {budget} ({pho*100}%)")
            
            with torch.no_grad():
                all_pseudo_labels, all_argmax_plabels,  all_conf,all_gt_labels = CSOT_PL(classifier, unlabeled_train_source_loader, num_class=type_num, batch_size=args.batch_size, 
                                                                                            budget=budget)
                all_pseudo_labels, all_argmax_plabels, all_conf,all_gt_labels = all_pseudo_labels.cpu(),all_argmax_plabels.cpu(),all_conf.cpu(),all_gt_labels.cpu()

        
                labeld_length = len(labeled_train_source_loader.dataset)
                assert labeld_length + len(all_argmax_plabels) == len(unlabeled_train_source_loader.dataset.labels)
                unlabeled_train_source_loader.dataset.labels[labeld_length:] = np.array(all_argmax_plabels)
                unlabeled_train_source_iter = ForeverDataIterator(unlabeled_train_source_loader)
                args.label_ratio = 1
                labeled_source_loader,_, _, _, _, _, _ = Prepare_scDataloader(args).getloader()
                labeled_train_source_iter = ForeverDataIterator(labeled_source_loader)


            train(labeled_train_source_iter,unlabeled_train_source_iter, train_target_iter, classifier, optimizer, lr_scheduler, epoch, type_num,train_target_loader,unlabeled_train_source_loader, args)

            acc1  = utils.validate(test_target_loader, classifier, args, device)
            torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
            print("test_acc1 = {:4.2f}".format(acc1))
            if best_acc1 < acc1:
                best_acc1 = acc1
                shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            print("best_acc1 = {:4.2f}".format(best_acc1))
        
    print("best_acc: {:4.2f}".format(best_acc1))
    

    
    logger.close()


    
def train(labeled_train_source_iter: ForeverDataIterator, unlabeled_train_source_iter: ForeverDataIterator,train_target_iter: ForeverDataIterator,
          model: sc_net, optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, type_num: int, train_target_loader, unlabeled_train_source_loader,args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Total Loss', ':6.2f')
    align_losses = AverageMeter('Align Loss', ':6.2f')
    un_losses_t = AverageMeter('Un Loss t', ':6.2f')
    tf_losses = AverageMeter('Transfer Loss s', ':6.2f')
    sf_losses_t = AverageMeter('Sf Loss t', ':6.2f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [ losses,tf_losses,sf_losses_t,align_losses,un_losses_t],
        prefix="Epoch: [{}]".format(epoch))


    n_t, c = len(train_target_loader.dataset.labels), type_num

    rollWindow = 5
    #theta_s = args.theta
    #inc_s = args.inc
    theta_t = args.theta
    inc_t = args.inc
    
    target_y = train_target_loader.dataset.labels
    target_y[target_y == -1] = 0
    target_p_Y = F.one_hot(torch.tensor(target_y),c)
    target_p_Y[target_y == -1] = 0
    confidence_t = deepcopy(target_p_Y)
    pre_correction_label_matrix_t = target_p_Y.clone().to(device)
    correction_label_matrix_t = target_p_Y.clone().to(device)
    f_record_t = torch.zeros([rollWindow, n_t, c]).to(device)
    confidence_t = confidence_t.to(device)
    confidence_t = confidence_t / confidence_t.sum(axis=1)[:, None]
    
    self_training_criterion = ConfidenceBasedSelfTrainingLoss(args.threshold).to(device)
    ts = TsallisEntropy(temperature=args.temperature, alpha=args.alpha)
    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(labeled_train_source_iter)[:2]
        x_t, labels_t,index_t = next(train_target_iter)[:3]
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        x_t_strong = add_noise(x_t).to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)
        index_t = index_t.to(device)
        
        # measure data loading time
        data_time.update(time.time() - end)

        # clear grad
        optimizer.zero_grad()

        # compute output
        #with torch.no_grad():
        f_t,y_t,_,_ = model(x_t)

        # cross entropy loss
        f_s,y_s,_,_ = model(x_s)
    
        cls_loss = F.cross_entropy(y_s, labels_s)

        
        _,y_t_strong,_,_ = model(x_t_strong)
        self_training_loss_target,_,_ = self_training_criterion(y_t_strong, y_t)
        self_training_loss_target = args.trade_off * self_training_loss_target
        
        
            
        ## unconfident samples for target samples
        soft_pseudo_labels_t = F.softmax(y_t.detach(), dim=1)
        confidence_t1, _ = soft_pseudo_labels_t.max(dim=1)
        soft_pseudo_labels_t = soft_pseudo_labels_t[confidence_t1 < args.threshold]
        index_t = index_t[confidence_t1 < args.threshold]
        p_Y_t = torch.zeros_like(soft_pseudo_labels_t)
        values_t,indices_t = torch.topk(soft_pseudo_labels_t, k = args.pll_topk,dim=1)
        p_Y_t.scatter_(1,indices_t,1)
        confidence_t[index_t, :] = p_Y_t.clone().detach()
        if len(soft_pseudo_labels_t)>0:
            L_ce_t, new_labels_t = proden_loss(y_t_strong[confidence_t1 < args.threshold], confidence_t[index_t, :].clone().detach(), None)
            confidence_t[index_t, :] = new_labels_t.clone().detach()
        else:
            L_ce_t = torch.from_numpy(np.array(0)).to(device)
        
        
        
        transfer_loss = ts(y_t)
     
        ## calculate class-related similarity
        weight = model.classifier[1].weight.detach()   ## C x D
        p_t_nograd = F.softmax(y_t.detach(), dim=1)
        f_target = F.normalize(torch.matmul(p_t_nograd, weight),dim=-1)
        p_s_nograd = F.softmax(y_s.detach(), dim=1)
        f_source = F.normalize(torch.matmul(p_s_nograd, weight),dim=-1)
        cls_sim =   torch.mm(f_target, f_source.transpose(0, 1)).detach()  
        # ## mixed sample
        zero_vec = -9e15 * torch.ones_like(cls_sim)
        cls_sim = torch.where(cls_sim > args.attn_threshold, cls_sim, zero_vec)
        cls_sim = F.softmax(cls_sim, dim=-1).detach()
        mixed_f_s = torch.mm(cls_sim,f_s).detach()
        
 
        align_loss = F.mse_loss(f_t,mixed_f_s)
        
        # measure accuracy and record loss
        loss = transfer_loss+ cls_loss + args.align_ratio*align_loss +  self_training_loss_target +  args.uncertain*L_ce_t
        loss.backward()
        losses.update(loss.item(), x_s.size(0))
        tf_losses.update(transfer_loss.item(), x_t.size(0))
        sf_losses_t.update(self_training_loss_target.item(), x_t.size(0))
        align_losses.update(align_loss.item(), x_s.size(0))
        un_losses_t.update(L_ce_t.item(), x_t.size(0))
        # compute gradient and do SGD step
        optimizer.step()
        

        f_record_t[epoch % rollWindow, :] = confidence_t
        if epoch >= args.warm_up and epoch % rollWindow == 0:

            temp_prob_matrix_t = f_record_t.mean(0)
            # label correction
            temp_prob_matrix_t = temp_prob_matrix_t / temp_prob_matrix_t.sum(dim=1).repeat(temp_prob_matrix_t.size(1),
                                                                                     1).transpose(0, 1)
            pre_correction_label_matrix_t = correction_label_matrix_t.clone()
            correction_label_matrix_t[temp_prob_matrix_t / torch.max(temp_prob_matrix_t, dim=1, keepdim=True)[0] < theta_t] = 0
            tmp_label_matrix_t = temp_prob_matrix_t * correction_label_matrix_t
            confidence_t = tmp_label_matrix_t / tmp_label_matrix_t.sum(dim=1).repeat(tmp_label_matrix_t.size(1), 1).transpose(0, 1)

            if theta_t < 0.4:
                if torch.sum(
                        torch.not_equal(pre_correction_label_matrix_t, correction_label_matrix_t)) < 0.0001 * n_t * c:
                    theta_t *= (inc_t + 1)
                         
        lr_scheduler.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def warmup(labeled_train_source_iter: ForeverDataIterator, model: sc_net, optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, type_num: int, args: argparse.Namespace):

    model.train()

    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(labeled_train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # clear grad
        optimizer.zero_grad()

        # cross entropy loss
        f_s,y_s,cluster_s,eqinv_s = model(x_s)
        cls_loss = F.cross_entropy(y_s, labels_s)

        loss =  cls_loss
        loss.backward()
        # compute gradient and do SGD step
        optimizer.step()
        #lr_scheduler.step()


def random_zeroing(data, zeroing_prob=0.1):
    B, _,D = data.size()
    num_zeros = int(zeroing_prob * D)
    zero_indices = torch.randperm(D)[:num_zeros]
    zeroed_data = data.clone()
    zeroed_data[:, :, zero_indices] = 0
    
    return zeroed_data

def add_noise(data, noise_level=0.001):
    noise = noise_level * torch.randn_like(data)
    return data + noise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multimodal single-cell data itegration')
    # dataset parameters
    parser.add_argument('--rna_data', default= ['./data/citeseq_control_rna.npz'])
    parser.add_argument('--rna_label', default= ['./data/citeseq_control_cellTypes.txt'])
    parser.add_argument('--atac_data', default= ['./data/asapseq_control_atac.npz'])
    parser.add_argument('--atac_label', default= ['./data/asapseq_control_cellTypes.txt'])
    parser.add_argument('--dataset', default= 'cite-asap')
    parser.add_argument('--label_ratio', type=float, default= 0.1 )
    
    # model parameters
    parser.add_argument('--trade-off', default= 0.1, type=float,help='the trade-off hyper-parameter for self training loss')
    parser.add_argument('--align_ratio', default=0.1, type=float,help='the trade-off hyper-parameter for align loss')
    parser.add_argument('--uncertain', default=0.1, type=float,help='the trade-off hyper-parameter for uncertain loss')
    # training parameters
    parser.add_argument('-theta', type=float, default=1e-3)
    parser.add_argument('-inc', type=float, default=1e-3)
    parser.add_argument('--pll_topk', default=4, type=int, help='candidate partial labels for uncertain samples')
    parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('-ub', '--unlabeled-batch-size', default=32, type=int, help='mini-batch size of unlabeled data (target domain) (default: 32)')
    parser.add_argument('--threshold', default=0.9, type=float, help='confidence threshold')
    parser.add_argument('--attn_threshold', default=0.9, type=float, help='atten threshold')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0004, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-3)', dest='weight_decay')
    parser.add_argument('--epochs', default=60, type=int, metavar='N')
    parser.add_argument('--warm_up', default=30, type=int, metavar='N')
    parser.add_argument('--temperature', default=2.0, type=float, help='parameter temperature scaling for entropy')
    parser.add_argument('--alpha', default= 1.9, type=float, help='the entropic index of Tsallis loss')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int, help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,  metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',  help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='ours_atac', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    
    if args.dataset =='cite-asap':
        args.rna_data = ['./data/citeseq_control_rna.npz'] 
        args.rna_label = ['./data/citeseq_control_cellTypes.txt'] 
        args.atac_data = ['./data/asapseq_control_atac.npz'] 
        args.atac_label = ['./data/asapseq_control_cellTypes.txt']  
    elif args.dataset == "scRNA_SMARTer-ATAC":
        args.rna_data = ['./data_MOp/YaoEtAl_RNA_scRNA_SMARTer_exprs.npz']
        args.rna_label = ['./data_MOp/YaoEtAl_RNA_scRNA_SMARTer_cellTypes.txt']
        args.atac_data = ['./data_MOp/YaoEtAl_ATAC_exprs.npz']
        args.atac_label = ['./data_MOp/YaoEtAl_ATAC_cellTypes.txt']
    elif args.dataset == "scRNA_SMARTer-snmC":
        args.rna_data = ['./data_MOp/YaoEtAl_RNA_scRNA_SMARTer_exprs.npz']
        args.rna_label = ['./data_MOp/YaoEtAl_RNA_scRNA_SMARTer_cellTypes.txt']
        args.atac_data = ['./data_MOp/YaoEtAl_snmC_exprs.npz']
        args.atac_label = ['./data_MOp/YaoEtAl_snmC_cellTypes.txt']
    elif args.dataset == "snRNA_10X_v3_A-ATAC":
        args.rna_data = ['./data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_exprs.npz']
        args.rna_label = ['./data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_cellTypes.txt']
        args.atac_data = ['./data_MOp/YaoEtAl_ATAC_exprs.npz']
        args.atac_label = ['./data_MOp/YaoEtAl_ATAC_cellTypes.txt']
    elif args.dataset == "snRNA_10X_v3_A-snmC":
        args.rna_data = ['./data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_exprs.npz']
        args.rna_label = ['./data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_cellTypes.txt']
        args.atac_data = ['./data_MOp/YaoEtAl_snmC_exprs.npz']
        args.atac_label = ['./data_MOp/YaoEtAl_snmC_cellTypes.txt']
    elif args.dataset == "snRNA_10X_v2-ATAC":
        args.rna_data = ['./data_MOp/YaoEtAl_RNA_snRNA_10X_v2_exprs.npz']
        args.rna_label = ['./data_MOp/YaoEtAl_RNA_snRNA_10X_v2_cellTypes.txt']
        args.atac_data = ['./data_MOp/YaoEtAl_ATAC_exprs.npz']
        args.atac_label = ['./data_MOp/YaoEtAl_ATAC_cellTypes.txt']           
    elif args.dataset == "snRNA_10X_v2-snmC":
        args.rna_data = ['./data_MOp/YaoEtAl_RNA_snRNA_10X_v2_exprs.npz']
        args.rna_label = ['./data_MOp/YaoEtAl_RNA_snRNA_10X_v2_cellTypes.txt']
        args.atac_data = ['./data_MOp/YaoEtAl_snmC_exprs.npz']
        args.atac_label = ['./data_MOp/YaoEtAl_snmC_cellTypes.txt']            
    elif args.dataset == "snRNA_SMARTer-ATAC":
        args.rna_data = ['./data_MOp/YaoEtAl_RNA_snRNA_SMARTer_exprs.npz']
        args.rna_label = ['./data_MOp/YaoEtAl_RNA_snRNA_SMARTer_cellTypes.txt']
        args.atac_data = ['./data_MOp/YaoEtAl_ATAC_exprs.npz']
        args.atac_label = ['./data_MOp/YaoEtAl_ATAC_cellTypes.txt'] 
    elif args.dataset == "snRNA_SMARTer-snmC":
        args.rna_data = ['./data_MOp/YaoEtAl_RNA_snRNA_SMARTer_exprs.npz']
        args.rna_label = ['./data_MOp/YaoEtAl_RNA_snRNA_SMARTer_cellTypes.txt']
        args.atac_data = ['./data_MOp/YaoEtAl_snmC_exprs.npz']
        args.atac_label = ['./data_MOp/YaoEtAl_snmC_cellTypes.txt']
    elif args.dataset == "scRNA_10X_v3-ATAC":
        args.rna_data = ['./data_MOp/YaoEtAl_RNA_scRNA_10X_v3_exprs.npz']
        args.rna_label = ['./data_MOp/YaoEtAl_RNA_scRNA_10X_v3_cellTypes.txt']
        args.atac_data = ['./data_MOp/YaoEtAl_ATAC_exprs.npz']
        args.atac_label = ['./data_MOp/YaoEtAl_ATAC_cellTypes.txt']
    elif args.dataset == "scRNA_10X_v3-snmC":
        args.rna_data = ['./data_MOp/YaoEtAl_RNA_scRNA_10X_v3_exprs.npz']
        args.rna_label = ['./data_MOp/YaoEtAl_RNA_scRNA_10X_v3_cellTypes.txt']
        args.atac_data = ['./data_MOp/YaoEtAl_snmC_exprs.npz']
        args.atac_label = ['./data_MOp/YaoEtAl_snmC_cellTypes.txt']
    elif args.dataset == "scRNA_10X_v2-ATAC":
        args.rna_data = ['./data_MOp/YaoEtAl_RNA_scRNA_10X_v2_exprs.npz']
        args.rna_label = ['./data_MOp/YaoEtAl_RNA_scRNA_10X_v2_cellTypes.txt']
        args.atac_data = ['./data_MOp/YaoEtAl_ATAC_exprs.npz']
        args.atac_label = ['./data_MOp/YaoEtAl_ATAC_cellTypes.txt']
    elif args.dataset == "scRNA_10X_v2-snmC":
        args.rna_data = ['./data_MOp/YaoEtAl_RNA_scRNA_10X_v2_exprs.npz']
        args.rna_label = ['./data_MOp/YaoEtAl_RNA_scRNA_10X_v2_cellTypes.txt']
        args.atac_data = ['./data_MOp/YaoEtAl_snmC_exprs.npz']
        args.atac_label = ['./data_MOp/YaoEtAl_snmC_cellTypes.txt']           
    main(args)
