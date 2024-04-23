import time
from collections import OrderedDict
import numpy as np
import torch
from utils.meters import AverageMeter
from torch.autograd import Variable
from cross_modality_rerank import re_ranking_cross

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def pairwise_distance(x, y):
   
    x, y = torch.tensor(x), torch.tensor(y)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)
        
    return dist_mat.numpy()


def cal_distmat(x, y):
    # compute dismat
    x = torch.tensor(x)
    y = torch.tensor(y)
    m, n = x.shape[0], y.shape[0]
    x = x.view(m, -1)
    y = y.view(n, -1)

    distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, x, y.t())
    return distmat.numpy()

## x_rgb=inputs, x_rgb_ca=inputs, x_ir=inputs


def extract_cnn_feature(model, inputs, mode, device):
    inputs = inputs.to(device)
    outputs_pool, outputs = model(inputs, inputs, mode=mode) ## 256 * 2048
    outputs_pool = outputs_pool.data.cpu()
    outputs = outputs.data.cpu()
    return outputs_pool, outputs


def extract_features_for_cluster(model, data_loader, print_freq=20, mode=None, device=0):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features_pool = OrderedDict() 
    features = OrderedDict() 
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (fnames, imgs, pids) in enumerate(data_loader):
            data_time.update(time.time() - end)
            outputs_pool, outputs = extract_cnn_feature(model, imgs, mode, device)

            for fname, output_pool, output, pid in zip(fnames, outputs_pool, outputs, pids):
                features_pool[fname] = output_pool
                features[fname] = output
                labels[fname] = int(pid)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features_pool, features, labels


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def test(args, model, ngall, nquery, gall_loader, query_loader, 
         query_label, gall_label, query_cam=None, gall_cam=None, test_mode=None):
    # switch to evaluation mode
    if args.dataset == 'sysu':
        test_mode = ['RGB', 'IR']
    elif args.dataset == 'regdb':
        test_mode = test_mode
        
    model.eval()
    
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    gall_pool = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.to(args.device))
            x_pool, feat = extract_cnn_feature(model, input, test_mode[0], args.device)
            flip_input = Variable(flip_input.to(args.device))
            flip_x_pool, flip_feat = extract_cnn_feature(model,flip_input, test_mode[0], args.device)
            feature_fc = (feat.detach() + flip_feat.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_pool[ptr:ptr + batch_num, :] = x_pool.detach().cpu().numpy()
            gall_feat[ptr:ptr + batch_num, :] = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    model.eval()


    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    query_pool = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.to(args.device))
            x_pool, feat = extract_cnn_feature(model, input, test_mode[1], args.device)
            flip_input = Variable(flip_input.to(args.device))
            flip_x_pool, flip_feat = extract_cnn_feature(model,flip_input, test_mode[1], args.device)
            feature_fc = (feat.detach() + flip_feat.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            query_pool[ptr:ptr + batch_num, :] = x_pool.detach().cpu().numpy()
            query_feat[ptr:ptr + batch_num, :] = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    # distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat = pairwise_distance(query_feat, gall_feat)

    if args.is_reranking == True:
        print("***********************Start reranking***********************")
        hetero_features_ir = query_feat
        hetero_features_rgb = gall_feat
        homo_features_ir = query_feat
        homo_features_rgb = gall_feat

        distmat_ir_rgb = cal_distmat(hetero_features_ir, hetero_features_rgb)
        distmat_rgb_rgb = cal_distmat(homo_features_rgb, homo_features_rgb)
        distmat_ir_ir = cal_distmat(homo_features_ir, homo_features_ir)

        if args.dataset == 'sysu':
            k_for_cmrr = 15
        elif args.dataset == 'regdb':
            k_for_cmrr = 15

        rerank_dist, _ = re_ranking_cross(distmat_ir_rgb, distmat_ir_rgb.T, 
                                        distmat_ir_ir, distmat_rgb_rgb,
                                        hetero_features_ir, hetero_features_rgb, 
                                        k=k_for_cmrr,eta_value=0.1)
        distmat = rerank_dist
        print("***********************Finish reranking***********************")
    
    # evaluation
    print('eval feat after batchnorm')

    if args.dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
    elif args.dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))
    return cmc, mAP, mINP





def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        
        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP


def eval_regdb(distmat, q_pids, g_pids, max_rank=20):
    """
    Evaluation with RegDB metric.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2 * np.ones(num_g).astype(np.int32)

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP
