import numpy as np
import torch
from torch import nn
import argparse
from sklearn.manifold import TSNE
from SYSU import SYSUMM01
from dataset import dataset_for_feature_extractor
import os.path as osp
import torch.utils.data as data
from model.network import BaseResNet
from utils.serialization import copy_state_dict, load_checkpoint
from evaluator import extract_features_for_cluster
import collections


def main():
    
    model_dirs = ['baseline/baseline_for_sysu', 'sysu_wo_OCLR/theta0.7alpha0.2', 'sysu_pretrain_1/theta0.7alpha0.2', 'OTLA_W_OCLR/theta1.0alpha0.05']
    save_feature = False


    args = parser.parse_args()
    img_h=288
    img_w=144
    data_dir = 'hlf/1_ReID_data'

    data_dir = osp.join(args.data_dir, 'SYSU-MM01')
    dataset = SYSUMM01(args, args.data_dir) 
    assign_RGB_set = dataset_for_feature_extractor(dataset.train_rgb, img_h=img_h, img_w=img_w)
    assign_IR_set = dataset_for_feature_extractor(dataset.train_ir, img_h=img_h, img_w=img_w)
    assign_RGB_loader = data.DataLoader(assign_RGB_set, batch_size=256, num_workers=4, drop_last=False)
    assign_IR_loader = data.DataLoader(assign_IR_set, batch_size=256, num_workers=4, drop_last=False)

    if save_feature == True:
        print('==> Building model..')
        main_net = BaseResNet(args, class_num=0, non_local='off', gm_pool='on', per_add_iters=args.per_add_iters)
        main_net.to(args.device)
        device_ids=[args.device, args.device + 1]
        main_net = nn.DataParallel(main_net, device_ids=device_ids)

        for i, model_dir in enumerate(model_dirs):
            checkpoint = load_checkpoint(osp.join(osp.dirname(osp.abspath(__file__)), 
                        'logs', model_dir, 'checkpoint.pth.tar'))
            copy_state_dict(checkpoint['state_dict'], main_net.module, strip='module.')
            
            _, features_RGB, gt_labels_RGB = extract_features_for_cluster(main_net, assign_RGB_loader, mode='RGB', device=args.device)
            _, features_IR, gt_labels_IR = extract_features_for_cluster(main_net, assign_IR_loader, mode='IR', device=args.device)
            features_RGB = torch.cat([features_RGB[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_rgb)], 0)
            features_IR = torch.cat([features_IR[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_ir)], 0)
            gt_labels_RGB = torch.cat([torch.tensor(gt_labels_RGB[f]).unsqueeze(0) for f, _, _ in sorted(dataset.train_rgb)])
            gt_labels_IR = torch.cat([torch.tensor(gt_labels_IR[f]).unsqueeze(0) for f, _, _ in sorted(dataset.train_ir)])

            np.save(osp.join(osp.dirname(osp.abspath(__file__)), '1_features', str(i+1), 'rgb.npy'), features_RGB)
            np.save(osp.join(osp.dirname(osp.abspath(__file__)), '1_features', str(i+1), 'ir.npy'), features_IR)

        np.save(osp.join(osp.dirname(osp.abspath(__file__)), '1_labels', 'rgb.npy'), gt_labels_RGB)
        np.save(osp.join(osp.dirname(osp.abspath(__file__)), '1_labels', 'ir.npy'), gt_labels_IR)

    else:
        features_RGB = []
        features_IR = []
        for i, model_dir in enumerate(model_dirs):
            features_RGB.append(np.load(osp.join(osp.dirname(osp.abspath(__file__)), '1_features', str(i+1), 'rgb.npy')))
            features_IR.append(np.load(osp.join(osp.dirname(osp.abspath(__file__)), '1_features', str(i+1), 'ir.npy')))

        gt_labels_RGB = np.load(osp.join(osp.dirname(osp.abspath(__file__)), '1_labels', 'rgb.npy'))
        gt_labels_IR = np.load(osp.join(osp.dirname(osp.abspath(__file__)), '1_labels', 'ir.npy'))

    # centers_RGB, features_RGB, gt_labels_RGB, labels_RGB, pseudo_dataset_RGB, dist_rgb, _ =\
    #         dbscan_cluster(args, features_RGB, gt_labels_RGB, dataset.train_rgb, cluster)  
    # centers_IR, features_IR, gt_labels_IR, labels_IR, pseudo_dataset_IR, dist_ir, _ =\
    #         dbscan_cluster(args, features_IR, gt_labels_IR, dataset.train_ir, cluster) 
    
    # features_RGB = torch.cat([feature.unsqueeze(0) for feature in features_RGB], dim=0) 
    # features_IR = torch.cat([feature.unsqueeze(0) for feature in features_IR], dim=0) 

    uni_labels = np.unique(gt_labels_RGB)
    ids = [uni_labels[1], uni_labels[4], uni_labels[6], uni_labels[13], uni_labels[34], uni_labels[85], uni_labels[2]]
    ids_dict = collections.defaultdict()
    for i, id in enumerate(ids):
        ids_dict[id] = i 
    # ids = [uni_labels[28], uni_labels[40], uni_labels[77], uni_labels[94], uni_labels[135]]
    rgb_idx = []
    ir_idx = []
    for id in ids:
        labels_rgb = np.where(gt_labels_RGB == id)[0]
        labels_ir = np.where(gt_labels_IR == id)[0]
        for label in labels_rgb:
            rgb_idx.append(label)
        for label in labels_ir:
            ir_idx.append(label)

    import matplotlib.pyplot as plt
    from sklearn import manifold, datasets

    label_size = 24
    shape_size = 18
    marker1, marker2 = 'o', 'x'
    num_rgb, num_ir = len(rgb_idx), len(ir_idx)
    # plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.figure(figsize=(14, 14))
    #   fig.tight_layout()#调整整体空白
    plt.subplots_adjust(wspace =0.04, hspace =0.13)#调整子图间距

    # legend_elements = [Line2D([0], [0], marker=marker1, c='none', label='Visible modality', linewidth=0, 
    #                       markeredgecolor='black', markersize=shape_size/3.5),
    #                 Line2D([0], [0], marker=marker2, c='none', label='Infrared modality', linewidth=0,
    #                       markeredgecolor='black', markersize=shape_size/3.5),]
    
    for i in range(4):

        if i >= 0:
        
            print('Start t-SNE')
            ts = TSNE(n_components=2)
            X = ts.fit_transform(np.vstack((features_RGB[i][rgb_idx], features_IR[i][ir_idx])))
            x_min, x_max = np.min(X, 0), np.max(X, 0)
            X = (X - x_min) / (x_max - x_min)
            x1, x2 = X[:, 0], X[:, 1]
            print('Finish t-SNE')

            plt.subplot(2, 2, i + 1)

            for j, (x, y) in enumerate(zip(x1, x2)):
                if j < num_rgb:
                    plt.scatter(x, y,
                            c=plt.cm.Set2(ids_dict[gt_labels_RGB[rgb_idx[j]]]/7.), edgecolors='none',
                            s=shape_size, alpha = 1.0, marker=marker1)
                else:
                    plt.scatter(x, y,
                            c=plt.cm.Set2(ids_dict[gt_labels_IR[ir_idx[j - num_rgb]]]/7.), edgecolors='none',
                            s=shape_size, alpha = 1.0, marker=marker2)

            plt.xticks([])
            plt.yticks([])

            if i == 0:
                plt.title('(a) Baseline', fontsize=label_size)
            elif i == 1:
                plt.title('(b) MIRL w/ MULT', fontsize=label_size)
            elif i == 2:
                plt.title('(c) MIRL + OCLR w/ MULT', fontsize=label_size)
            else:
                plt.title('(d) MIRL + OCLR w/ DOTLA', fontsize=label_size)

        # plt.legend(handles=legend_elements, loc='upper left')
    
    plt.savefig('hlf/LP_iter/1_tsne/t-sne-test.pdf', bbox_inches='tight')

    print('Finish saving')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="assignment main train")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    ## default
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]') ### DATASET
    parser.add_argument('--mode', default='all', type=str, help='sysu:all/indoor regdb:visibletothermal')
    ### sysu: all indoor
    ### regdb: visibletothermal
    parser.add_argument('--epochs', default=90)
    parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
    parser.add_argument('--test-batch', default=256, type=int,
                    metavar='tb', help='testing batch size')
    parser.add_argument('--batch-size', default=12, type=int,
                    metavar='B', help='training batch size')
    parser.add_argument('--num_pos', default=12, type=int, 
                    help='num of pos per identity in each modality')
    parser.add_argument('--print-step', default=50, type=int)
    parser.add_argument('--eval-step', default=1, type=int)

    parser.add_argument('--stage-one', default=40, type=int)
    parser.add_argument('--stage-two', default=40, type=int)
    parser.add_argument('--stage-three', default=90, type=int)
    
    ## cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN") ## 0.6 for sysu and 0.3 for regdb
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    
    ## network  
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--pool-dim', default=2048)
    parser.add_argument('--per-add-iters', default=1, help='param for GRL')
    parser.add_argument('--lr', default=0.00035, help='learning rate for main net')
    parser.add_argument('--optim', default='adam', help='optimizer')
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--train-iter', type=int, default=300) ## 100 for regdb and 300 for sysu
    parser.add_argument('--pretrained', type=bool, default=False)
    
    ## memory
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--momentum-cross', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--use-hard', default=False)
    
    parser.add_argument('--alpha', default=0.20)
    parser.add_argument('--theta', default=0.70)
    parser.add_argument('--device', type=int, default=0)
    ## 0.2 0.7


    ## path
    
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='hlf/1_ReID_data')
    # parser.add_argument('--logs-dir', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, 'logs/sysu_final_2'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/sysu_agw'))
    main()