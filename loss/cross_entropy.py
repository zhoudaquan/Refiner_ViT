import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        # import pdb;pdb.set_trace()
        if not N==N_rep:
            target = target.repeat(N_rep//N,1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class SoftTargetCrossEntropyCosReg(nn.Module):

    def __init__(self, n_comn=2):
        super(SoftTargetCrossEntropyCosReg, self).__init__()
        self.dis_fn = torch.nn.CosineSimilarity(dim=1)
        self.n_comn = n_comn

    def forward(self, x, target, atten=None):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        cos_loss = 0
        if atten is not None:
            for i in range(self.n_comn):
                cos_loss += self.dis_fn(atten[i], atten[i+1])
        # import pdb;pdb.set_trace()
        return loss.mean() + 0.1 * cos_loss.mean()/self.n_comn

class RelabelSoftTargetCrossEntropy(nn.Module):

    def __init__(self, n_comn=2):
        super(RelabelSoftTargetCrossEntropy, self).__init__()
        self.n_comn = n_comn

    def forward(self, x, target, atten=None):
        N_rep = x.shape[0]
        N = target.shape[0]
        # import pdb;pdb.set_trace()
        # cal cos similarity loss
        cos_loss = 0
        if atten is not None:
            for i in range(self.n_comn):
                cos_loss += self.dis_fn(atten[i], atten[i+1])

        if not N==N_rep:
            target = target.repeat(N_rep//N,1)
        if len(target.shape)==3 and target.shape[-1]==2:
            ground_truth=target[:,:,0]
            target = target[:,:,1]
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1) + 0.1 * cos_loss.mean()/self.n_comn
        return loss.mean()

class RelabelCrossEntropy(nn.Module):
    """
    Relabel dense loss.
    """
    def __init__(self, dense_weight=1.0, cls_weight = 1.0, mixup_active=True, smoothing=0.1,
        classes = 1000, n_comn=2):
        """
        Constructor Relabel dense loss.
        """
        super(RelabelCrossEntropy, self).__init__()


        self.CE = SoftTargetCrossEntropy()

        self.n_comn = n_comn
        self.dis_fn = nn.CosineSimilarity(dim=1)

        self.dense_weight = dense_weight
        self.smoothing = smoothing
        self.mixup_active = mixup_active
        self.classes = classes
        self.cls_weight = cls_weight
        assert dense_weight+cls_weight>0


    def forward(self, x, target, atten = None):

        output, aux_output, bb = x
        bbx1, bby1, bbx2, bby2 = bb

        B,N,C = aux_output.shape
        # B,C,H,W = target.shape
        if len(target.shape)==2:
            target_cls=target
            # target_aux = target.unsqueeze(2).repeat(1,1,N).transpose(1,2).reshape(B*N,C)
            target_aux = target.repeat(1,N).reshape(B*N,C)
        else:
            ground_truth = target[:,:,0]
            target_cls = target[:,:,1]
            target_aux = target[:,:,2:]
            target_aux = target_aux.transpose(1,2).reshape(-1,C)
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / N)
        if lam<1:
            target_cls = lam*target_cls + (1-lam)*target_cls.flip(0)

        
        aux_output = aux_output.reshape(-1,C)
        

        loss_cls = self.CE(output, target_cls)
        loss_aux = self.CE(aux_output, target_aux)

        cos_loss = 0
        if atten is not None:
            for i in range(self.n_comn):
                cos_loss += self.dis_fn(atten[i], atten[i+1])
        cos_loss = 0 if atten is None else 0.1 * cos_loss.mean()/self.n_comn
        return self.cls_weight*loss_cls+self.dense_weight* loss_aux + cos_loss


class RelabelPooledCrossEntropy(nn.Module):
    """
    Relabel dense loss.
    """
    def __init__(self, dense_weight=1.0, cls_weight = 1.0, mixup_active=True, smoothing=0.1,
        classes = 1000):
        """
        Constructor Relabel dense loss.
        """
        super(RelabelPooledCrossEntropy, self).__init__()


        self.CE = SoftTargetCrossEntropy()

        self.dense_weight = dense_weight
        self.smoothing = smoothing
        self.mixup_active = mixup_active
        self.classes = classes
        self.cls_weight = cls_weight
        assert dense_weight+cls_weight>0


    def forward(self, x, target):

        output, aux_output, bb = x
        bbx1, bby1, bbx2, bby2 = bb

        B,N,C = aux_output.shape
        # B,C,H,W = target.shape

        target_cls=target

        target_aux = target
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / N)
        if lam<1:
            target_cls = lam*target_cls + (1-lam)*target_cls.flip(0)
    
        aux_output = aux_output.mean(1)    
        loss_cls = self.CE(output, target_cls)
        loss_aux = self.CE(aux_output, target_aux)


        return self.cls_weight*loss_cls+self.dense_weight* loss_aux
