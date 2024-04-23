import numpy as np
import torch
import argparse
import os.path as osp
from matplotlib.pyplot import MultipleLocator

from hlf.MULT_master.cross_association import pairwise_distance



def main():

    ## 1- 黄蓝配色 color1是前面的颜色
    color1 = '#edac5d'
    color2 = '#2378B7'

    ## 2- 玄青 橘红
    # color2 = '#3d3b4f'
    # color1 = '#ff7500'

    ##
    # color1 = '#c3a315'
    # color2 = '#314396'

    edgecolor2 = 'black'
    edgecolor1 = 'black'
    
    model_dirs = ['baseline/baseline_for_sysu', 'sysu_wo_OCLR/theta0.7alpha0.2',
                'sysu-final/theta0.7alpha0.2', 'OT_w_OCLR/theta0.7alpha0.2']

    features_RGB = []
    features_IR = []
    for i, model_dir in enumerate(model_dirs):
        features_RGB.append(torch.tensor(np.load(osp.join(osp.dirname(osp.abspath(__file__)), '1_features', str(i+1), 'rgb.npy'))))
        features_IR.append(torch.tensor(np.load(osp.join(osp.dirname(osp.abspath(__file__)), '1_features', str(i+1), 'ir.npy'))))

    gt_labels_RGB = torch.tensor(np.load(osp.join(osp.dirname(osp.abspath(__file__)), '1_labels', 'rgb.npy')))
    gt_labels_IR = torch.tensor(np.load(osp.join(osp.dirname(osp.abspath(__file__)), '1_labels', 'ir.npy')))

    mask = torch.as_tensor(gt_labels_RGB.expand(gt_labels_IR.shape[0], gt_labels_RGB.shape[0])\
    .eq(gt_labels_IR.expand(gt_labels_RGB.shape[0], gt_labels_IR.shape[0]).t()), dtype=torch.float)
    pos_idx1, pos_idx2 = torch.where(mask[:10000, :100000] == 1)
    neg_idx1, neg_idx2 = torch.where(mask[:10000, :10000] == 0)

    import matplotlib.pyplot as plt
    from sklearn import manifold, datasets
    import seaborn as sns

    label_size = 18
    shape_size = 20
    marker1, marker2 = 'o', 'D'
    # num_rgb, num_ir = len(rgb_idx), len(ir_idx)

    plt.figure(figsize=(16, 12)) ## (16, 5)
    #   fig.tight_layout()#调整整体空白
    plt.subplots_adjust(wspace =0.185, hspace =0.28)#调整子图间距

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        dist = pairwise_distance(features_IR[i], features_RGB[i])
        pos_dist = dist[pos_idx1, pos_idx2][30000:90000]
        neg_dist = dist[neg_idx1, neg_idx2][30000:90000]

        plt.hist(neg_dist, bins=50, color=color2, alpha=0.8, edgecolor=edgecolor2, density=True)
        plt.hist(pos_dist, bins=50, color=color1, alpha=0.8, edgecolor=edgecolor1, density=True)
        # plt.hist(neg_dist, bins=60, color=color2, alpha=0.9, edgecolor=color2)
        # plt.hist(pos_dist, bins=60, color=color1, alpha=0.9, edgecolor=color1)

        # plt.xticks([])
        # plt.yticks([])
        plt.xlim((0.35, 2.54))
        plt.ylim((0, 2.20))
        plt.xticks(size=20)
        plt.yticks(size=20)
        x_major_locator=MultipleLocator(0.5)
        y_major_locator=MultipleLocator(0.5)
        ax=plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)

        plt.ylabel('Density', fontsize=22)
        plt.xlabel('Euclidean Distance', fontsize=22)
        plt.legend(['V-I negative', 'V-I positive'],loc = "upper center", fontsize=20)


        if i == 0:
            plt.title('(a) Baseline', fontsize=22)
        elif i == 1:
            plt.title('(b) MIRL w/ MULT', fontsize=22)
        elif i == 2:
            plt.title('(c) MIRL + OCLR w/ MULT', fontsize=22)
        elif i == 3:
            plt.title('(d) MIRL + OCLR w/ DOTLA', fontsize=22)

    # plt.savefig('hlf/LP_iter/plot_distribution.jpg', bbox_inches='tight')
    plt.savefig('hlf/LP_iter/plot_distribution.pdf', bbox_inches='tight')


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
    parser.add_argument('--device', type=int, default=2)
    ## 0.2 0.7


    ## path
    
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='hlf/1_ReID_data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/test'))
    main()