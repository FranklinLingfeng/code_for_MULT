import torch
from utils.meters import AverageMeter
from optimizer import adjust_learning_rate
import time
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

def pairwise_distance(x, y):
   
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)
        
    return dist_mat
    

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()
        self.ratio = 0.2

    def forward(self, pred, target):

        B = pred.shape[0]
        pred = torch.softmax(pred, dim=1)
        target = torch.softmax(target / self.ratio, dim=1)

        loss = (-pred.log() * target).sum(1).sum() / B

        return loss



class AssignTrainer(object):
    def __init__(self, args, encoder, batch_size, num_pos):
        
        super(AssignTrainer, self).__init__()
        self.encoder = encoder
        
        self.m_v = None
        self.m_r = None
        self.m_va = None
        self.m_ra = None
        self.m_rcm = None
        self.m_vcm = None
        
        self.temp = 0.05
                
        self.mask_matrix = None
        
        self.criterion_dis = nn.BCELoss()
        
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.transform_to_image = transforms.Compose([
             transforms.ToPILImage()
        ])
        
        # self.theta = args.theta
        
        self.sce_loss = SoftCrossEntropyLoss()

    def train(self, args, epoch, trainloader, optimizer, device, stage=None):
        current_lr = adjust_learning_rate(args, optimizer, epoch)

        batch_time = AverageMeter()
        contrast_rgb = AverageMeter()
        contrast_ir = AverageMeter()
        contrast_cross_rgb = AverageMeter()
        contrast_cross_ir = AverageMeter()
        OCLR_loss = AverageMeter()      
        Dis_loss = AverageMeter()  

        self.encoder.train()
        end = time.time()
                        
        print('epoch:{:5d}'.format(epoch))

        for batch_idx, (img10, img11, img2, label1, label2, \
                        cm_label1, cm_label2, y1, y2, cm_y1, cm_y2) in enumerate(trainloader):
            
            input10 = Variable(img10.to(device))
            input11 = Variable(img11.to(device))
            input2 = Variable(img2.to(device))    

            label1 = Variable(label1.to(device))
            label2 = Variable(label2.to(device))
            cm_label1 = Variable(cm_label1.to(device))
            cm_label2 = Variable(cm_label2.to(device))

            y1 = Variable(y1.to(device))
            y2 = Variable(y2.to(device))
            cm_y1 = Variable(cm_y1.to(device))
            cm_y2 = Variable(cm_y2.to(device))

            if stage == 'single':

                input1 = torch.cat((input10, input11), dim=0)
            
                feat_1, feat_2, dis_1, dis_2 = self.encoder(input1, input2)  
                feat_rgb0 = feat_1[:feat_1.shape[0]//2]
                feat_rgb1 = feat_1[feat_1.shape[0]//2:]
                feat_ir = feat_2
                dis_rgb0 = dis_1[:feat_1.shape[0]//2]
                dis_rgb1 = dis_1[feat_1.shape[0]//2:]
                dis_ir = dis_2

                output_v_v = self.m_v(feat_rgb0, label1)
                output_a_v = self.m_v(feat_rgb1, label1)
                output_r_r = self.m_r(feat_ir, label2)

                loss_contrast_rgb = F.cross_entropy(output_v_v, label1) + F.cross_entropy(output_a_v, label1)
                loss_contrast_ir = F.cross_entropy(output_r_r, label2)

                # B = dis_rgb0.shape[0]
                # dis_pred = torch.cat((dis_rgb0, dis_rgb1, dis_ir), dim=0)
                # dis_label = torch.cat((torch.ones(B), torch.ones(B), torch.zeros(B)), dim=0).to(device, dtype=torch.int64)
                # loss_dis = F.cross_entropy(dis_pred, dis_label)

                loss_dis = torch.tensor(0)
                loss_all = loss_contrast_rgb + loss_contrast_ir + loss_dis

                contrast_rgb.update(loss_contrast_rgb.item())
                contrast_ir.update(loss_contrast_ir.item())
                Dis_loss.update(loss_dis.item()) 

            elif stage == 'cross':

                if epoch % 2 == 0:
                
                    input1 = torch.cat((input10, input11), dim=0)
                    feat_1, feat_2, dis_1, dis_2 = self.encoder(input1, input2)  
                    feat_rgb0 = feat_1[:feat_1.shape[0]//2]
                    feat_rgb1 = feat_1[feat_1.shape[0]//2:]
                    feat_ir = feat_2
                    
                    output_v_v = self.m_v(feat_rgb0, label1)
                    output_a_v = self.m_v(feat_rgb1, label1)
                    output_r_rcm = self.m_rcm(feat_ir, cm_label2)

                    output_r_r = self.m_r(feat_ir, label2)
                    
                    output_v_va = self.m_va(feat_rgb0, label1)
                    output_a_va = self.m_va(feat_rgb1, label1)
                    output_r_va = self.m_va(feat_ir, cm_label2) 

                    feat_rgb0 = F.normalize(feat_rgb0, dim=1)
                    feat_rgb1 = F.normalize(feat_rgb1, dim=1)
                    feat_ir = F.normalize(feat_ir, dim=1)

                    output_v_rcm = feat_rgb0.mm(self.m_rcm.features.t()) / self.temp
                    output_a_rcm = feat_rgb1.mm(self.m_rcm.features.t()) / self.temp
                    output_r_v = feat_ir.mm(self.m_v.features.t()) / self.temp

                    loss_ms_rgb = F.cross_entropy(output_v_v, y1) + F.cross_entropy(output_a_v, y1)
                    loss_ms_ir = F.cross_entropy(output_r_r, y2) 
                    loss_ms_ir += F.cross_entropy(output_r_rcm, cm_y2)
                    # loss_ms_ir = F.cross_entropy(output_r_rcm, cm_y2)

                    loss_ma_rgb = F.cross_entropy(output_v_va, y1) + F.cross_entropy(output_a_va, y1)
                    loss_ma_ir = F.cross_entropy(output_r_va, cm_y2)

                    loss_oclr = self.sce_loss(output_v_va, output_a_v) +\
                            self.sce_loss(output_a_va, output_v_v) +\
                            self.sce_loss(output_r_va, output_r_rcm) +\
                            self.sce_loss(output_v_va, output_a_rcm) +\
                            self.sce_loss(output_a_va, output_v_rcm) +\
                            self.sce_loss(output_r_va, output_r_v)
                    
                                        
                elif epoch % 2 == 1:
                    
                    input1 = torch.cat((input10, input11), dim=0)
                    feat_1, feat_2, dis_1, dis_2 = self.encoder(input1, input2)  
                    feat_rgb0 = feat_1[:feat_1.shape[0]//2]
                    feat_rgb1 = feat_1[feat_1.shape[0]//2:]
                    feat_ir = feat_2
                    
                    output_v_vcm = self.m_vcm(feat_rgb0, cm_label1)
                    output_a_vcm = self.m_vcm(feat_rgb1, cm_label1)
                    output_r_r = self.m_r(feat_ir, label2)
                    
                    output_v_v = self.m_v(feat_rgb0, label1)
                    output_a_v = self.m_v(feat_rgb1, label1)

                    output_v_ra = self.m_ra(feat_rgb0, cm_label1)
                    output_a_ra = self.m_ra(feat_rgb1, cm_label1)
                    output_r_ra = self.m_ra(feat_ir, label2)

                    feat_rgb0 = F.normalize(feat_rgb0, dim=1)
                    feat_rgb1 = F.normalize(feat_rgb1, dim=1)
                    feat_ir = F.normalize(feat_ir, dim=1)

                    output_v_r = feat_rgb0.mm(self.m_r.features.t()) / self.temp
                    output_a_r = feat_rgb1.mm(self.m_r.features.t()) / self.temp
                    output_r_vcm = feat_ir.mm(self.m_vcm.features.t()) / self.temp
                    
                    loss_ms_rgb = F.cross_entropy(output_v_v, y1) + F.cross_entropy(output_a_v, y1)
                    loss_ms_rgb += F.cross_entropy(output_v_vcm, cm_y1) + F.cross_entropy(output_a_vcm, cm_y1)
                    # loss_ms_rgb = F.cross_entropy(output_v_vcm, cm_y1) + F.cross_entropy(output_a_vcm, cm_y1)
                    loss_ms_ir = F.cross_entropy(output_r_r, y2)

                    loss_ma_rgb = F.cross_entropy(output_v_ra, cm_y1) + F.cross_entropy(output_a_ra, cm_y1)
                    loss_ma_ir = F.cross_entropy(output_r_ra, y2)

                    loss_oclr = self.sce_loss(output_v_ra, output_a_vcm) +\
                            self.sce_loss(output_a_ra, output_v_vcm) +\
                            self.sce_loss(output_r_ra, output_r_r) +\
                            self.sce_loss(output_v_ra, output_a_r) +\
                            self.sce_loss(output_a_ra, output_v_r) +\
                            self.sce_loss(output_r_ra, output_r_vcm)
                    
                else:
                    output_a_v = self.m_v(feat_rgb1, label1)
                    output_v_v = self.m_v(feat_rgb0, label1)
                    output_r_r = self.m_r(feat_ir, label2)

                    loss_ms_rgb = F.cross_entropy(output_v_v, label1) + F.cross_entropy(output_a_v, label1)
                    loss_ms_ir = F.cross_entropy(output_r_r, label2)
                    loss_ma_rgb, loss_ma_ir, loss_oclr = torch.tensor(0), torch.tensor(0), torch.tensor(0)
                    
                loss_ms = loss_ms_rgb + loss_ms_ir
                loss_ma = loss_ma_rgb + loss_ma_ir
                loss_all = loss_ms + loss_ma + loss_oclr
                # loss_all = loss_ms + loss_ma

                loss_dis = torch.tensor(0)
                                          
                contrast_rgb.update(loss_ms_rgb.item())
                contrast_ir.update(loss_ms_ir.item())   
                contrast_cross_rgb.update(loss_ma_rgb.item())
                contrast_cross_ir.update(loss_ma_ir.item())
                OCLR_loss.update(loss_oclr.item()) 
                Dis_loss.update(loss_dis.item()) 
                
                
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
         

            # print log
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (batch_idx + 1) % args.print_step == 0:
                if stage == 'single':
                    print('Epoch: [{}][{}/{}] '
                        'lr:{:.8f} '
                        'contrast_RGB: {contrast_rgb.val:.4f}({contrast_rgb.avg:.3f}) '
                        'contrast_IR: {contrast_ir.val:.4f}({contrast_ir.avg:.3f}) '
                        'Dis_loss: {Dis_loss.val:.4f}({Dis_loss.avg:.3f}) '
                                                        .format(
                                                        epoch, batch_idx, len(trainloader), current_lr,
                                                        contrast_rgb = contrast_rgb,
                                                        contrast_ir = contrast_ir,
                                                        Dis_loss = Dis_loss))
                elif stage == 'cross':
                    print('Epoch: [{}][{}/{}] '
                        'lr:{:.8f} '
                        'RGB-MSLoss: {contrast_rgb.val:.4f}({contrast_rgb.avg:.3f}) '
                        'IR-MSLoss: {contrast_ir.val:.4f}({contrast_ir.avg:.3f}) '
                        '\nRGB-MAloss: {contrast_cross_rgb.val:.4f}({contrast_cross_rgb.avg:.3f})'
                        'IR-MAloss: {contrast_cross_ir.val:.4f}({contrast_cross_ir.avg:.3f})'
                        'OCLR_loss: {OCLR_loss.val:.4f}({OCLR_loss.avg:.3f})'
                        'Dis_loss: {Dis_loss.val:.4f}({Dis_loss.avg:.3f}) '.format(
                                                        epoch, batch_idx, len(trainloader), current_lr,
                                                        contrast_rgb = contrast_rgb,
                                                        contrast_ir = contrast_ir,
                                                        contrast_cross_rgb = contrast_cross_rgb,
                                                        contrast_cross_ir = contrast_cross_ir,
                                                        OCLR_loss = OCLR_loss,
                                                        Dis_loss = Dis_loss))   
                
                else:
                    print('-----Wrong stage-----')


            
            
            
