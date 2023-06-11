import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *
from models.modules import *
# from Refinement_new import network
import torch.nn.functional as F

class PASMnet(nn.Module):
    def __init__(self):
        super(PASMnet, self).__init__()
        ###############################################################
        ## scale     #  1  #  1/2  #  1/4  #  1/8  #  1/16  #  1/32  ##
        ## channels  #  16 #  32   #  64   #  96   #  128   #  160   ##
        ###############################################################

        # Feature Extraction
        self.feature_ex = RestNet18()
        # self.hourglass = Hourglass([32, 64, 96, 128, 160])

        # Cascaded Parallax-Attention Module
        self.cas_pam = CascadedPAM([128, 96, 64])

        # Output Module
        self.output = Output()

        # Disparity Refinement
        # self.refine = Refinement_new()
        self.refine = Refinement([64, 96, 128, 160, 160, 128, 96, 64, 32, 16])

    def forward(self, x_left, x_right, max_disp=0):
        b, _, h, w = x_left.shape

        # Feature Extraction
        (fea_left_s1, fea_left_s2, fea_left_s3), fea_refine = self.feature_ex(x_left)
        (fea_right_s1, fea_right_s2, fea_right_s3), _ = self.feature_ex(x_right)
        
        # (fea_left_s1, fea_left_s2, fea_left_s3), fea_refine = self.hourglass(x_left)
        # (fea_right_s1, fea_right_s2, fea_right_s3), _       = self.hourglass(x_right)

        # Cascaded Parallax-Attention Module
        cost_s1, cost_s2, cost_s3 = self.cas_pam([fea_left_s1, fea_left_s2, fea_left_s3],
                                                 [fea_right_s1, fea_right_s2, fea_right_s3])

        # Output Module
        if self.training:
            disp_s1, att_s1, att_cycle_s1, valid_mask_s1 = self.output(cost_s1, max_disp // 16)
            disp_s2, att_s2, att_cycle_s2, valid_mask_s2 = self.output(cost_s2, max_disp // 8)
            disp_s3, att_s3, att_cycle_s3, valid_mask_s3 = self.output(cost_s3, max_disp // 4)
        else:
            disp_s3 = self.output(cost_s3, max_disp // 4)

        # Disparity Refinement
        disp = self.refine(fea_refine, disp_s3)
        # disp = self.refine(disp_s3, x_left)

        if self.training:
            return disp, \
                   [att_s1, att_s2, att_s3], \
                   [att_cycle_s1, att_cycle_s2, att_cycle_s3], \
                   [valid_mask_s1, valid_mask_s2, valid_mask_s3]
        else:
            return disp


class BasicBlock(nn.Module):  #basic block for Conv2d
    def __init__(self,in_planes,planes,stride=1):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(planes)
        self.shortcut=nn.Sequential()
    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        out=F.relu(out)
        return out



# Resnet Module for Feature Extraction
class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(32, 32, 1),
                                    RestNetBasicBlock(32, 32, 1))  #1/2

        self.layer2 = nn.Sequential(RestNetDownBlock(32, 64, [2, 1]),
                                    RestNetBasicBlock(64, 64, 1))   #1/4

        self.layer3 = nn.Sequential(RestNetDownBlock(32, 96, [2, 1]),
                                    RestNetDownBlock(96, 96, [2, 1]),
                                    RestNetBasicBlock(96, 96, 1)) #1/8

        self.layer4 = nn.Sequential(RestNetDownBlock(32, 64, [2, 1]),
                                    RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetDownBlock(128, 128, [2, 1]), # 1/16
                                    RestNetBasicBlock(128, 128, 1))

        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        fea_D3 = self.layer2(out)
        fea_D2 = self.layer3(out)
        fea_D1 = self.layer4(out)
        # out = self.layer4(out)
        # out = self.avgpool(out)
        # out = out.reshape(x.shape[0], -1)
        # out = self.fc(out)
        # print("===feature_shape====", fea_D1.shape, fea_D2.shape, fea_D3.shape)
        return (fea_D1, fea_D2, fea_D3), fea_D3

# Hourglass Module for Feature Extraction
class Hourglass(nn.Module):
    def __init__(self, channels):
        super(Hourglass, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.E0 = EncoderB(1,           3, channels[0], downsample=True)               # scale: 1/2
        self.E1 = EncoderB(1, channels[0], channels[1], downsample=True)               # scale: 1/4
        self.E2 = EncoderB(1, channels[1], channels[2], downsample=True)               # scale: 1/8
        self.E3 = EncoderB(1, channels[2], channels[3], downsample=True)               # scale: 1/16
        self.E4 = EncoderB(1, channels[3], channels[4], downsample=True)               # scale: 1/32

        self.D0 = EncoderB(1, channels[4], channels[4], downsample=False)              # scale: 1/32
        self.D1 = DecoderB(1, channels[4] + channels[3], channels[3])                  # scale: 1/16
        self.D2 = DecoderB(1, channels[3] + channels[2], channels[2])                  # scale: 1/8
        self.D3 = DecoderB(1, channels[2] + channels[1], channels[1])                  # scale: 1/4

    def forward(self, x):
        fea_E0 = self.E0(x)                                                            # scale: 1/2
        fea_E1 = self.E1(fea_E0)                                                       # scale: 1/4
        fea_E2 = self.E2(fea_E1)                                                       # scale: 1/8
        fea_E3 = self.E3(fea_E2)                                                       # scale: 1/16
        fea_E4 = self.E4(fea_E3)                                                       # scale: 1/32

        fea_D0 = self.D0(fea_E4)                                                       # scale: 1/32
        fea_D1 = self.D1(torch.cat((self.upsample(fea_D0), fea_E3), 1))                # scale: 1/16
        fea_D2 = self.D2(torch.cat((self.upsample(fea_D1), fea_E2), 1))                # scale: 1/8
        fea_D3 = self.D3(torch.cat((self.upsample(fea_D2), fea_E1), 1))                # scale: 1/4

        # print("===feature_shape====", fea_D1.shape, fea_D2.shape, fea_D3.shape)
        return (fea_D1, fea_D2, fea_D3), fea_E1


# Cascaded Parallax-Attention Module
class CascadedPAM(nn.Module):
    def __init__(self, channels):
        super(CascadedPAM, self).__init__()
        self.stage1 = PAM_stage(channels[0])
        self.stage2 = PAM_stage(channels[1])
        self.stage3 = PAM_stage(channels[2])

        # bottleneck in stage 2
        self.b2 = nn.Sequential(
            nn.Conv2d(128 + 96, 96, 1, 1, 0, bias=True),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # bottleneck in stage 3
        self.b3 = nn.Sequential(
            nn.Conv2d(96 + 64, 64, 1, 1, 0, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )


    def forward(self, fea_left, fea_right):
        '''
        :param fea_left:    feature list [fea_left_s1, fea_left_s2, fea_left_s3]
        :param fea_right:   feature list [fea_right_s1, fea_right_s2, fea_right_s3]
        '''
        fea_left_s1, fea_left_s2, fea_left_s3 = fea_left
        fea_right_s1, fea_right_s2, fea_right_s3 = fea_right

        b, _, h_s1, w_s1 = fea_left_s1.shape
        b, _, h_s2, w_s2 = fea_left_s2.shape

        # stage 1: 1/16
        cost_s0 = [
            torch.zeros(b, h_s1, w_s1, w_s1).to(fea_right_s1.device),
            torch.zeros(b, h_s1, w_s1, w_s1).to(fea_right_s1.device)
        ]

        fea_left, fea_right, cost_s1 = self.stage1(fea_left_s1, fea_right_s1, cost_s0)

        # stage 2: 1/8
        fea_left = F.interpolate(fea_left, scale_factor=2, mode='bilinear')
        fea_right = F.interpolate(fea_right, scale_factor=2, mode='bilinear')
        fea_left = self.b2(torch.cat((fea_left, fea_left_s2), 1))
        fea_right = self.b2(torch.cat((fea_right, fea_right_s2), 1))

        cost_s1_up = [
            F.interpolate(cost_s1[0].view(b, 1, h_s1, w_s1, w_s1), scale_factor=2, mode='trilinear').squeeze(1),
            F.interpolate(cost_s1[1].view(b, 1, h_s1, w_s1, w_s1), scale_factor=2, mode='trilinear').squeeze(1)
        ]

        fea_left, fea_right, cost_s2 = self.stage2(fea_left, fea_right, cost_s1_up)

        # stage 3: 1/4
        fea_left = F.interpolate(fea_left, scale_factor=2, mode='bilinear')
        fea_right = F.interpolate(fea_right, scale_factor=2, mode='bilinear')
        fea_left = self.b3(torch.cat((fea_left, fea_left_s3), 1))
        fea_right = self.b3(torch.cat((fea_right, fea_right_s3), 1))

        cost_s2_up = [
            F.interpolate(cost_s2[0].view(b, 1, h_s2, w_s2, w_s2), scale_factor=2, mode='trilinear').squeeze(1),
            F.interpolate(cost_s2[1].view(b, 1, h_s2, w_s2, w_s2), scale_factor=2, mode='trilinear').squeeze(1)
        ]

        fea_left, fea_right, cost_s3 = self.stage3(fea_left, fea_right, cost_s2_up)

        return [cost_s1, cost_s2, cost_s3]


class PAM_stage(nn.Module):
    def __init__(self, channels):
        super(PAM_stage, self).__init__()
        self.pab1 = PAB(channels)
        self.pab2 = PAB(channels)
        self.pab3 = PAB(channels)
        self.pab4 = PAB(channels)

    def forward(self, fea_left, fea_right, cost):
        fea_left, fea_right, cost = self.pab1(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab2(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab3(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab4(fea_left, fea_right, cost)

        return fea_left, fea_right, cost


# Disparity Refinement Module
class Refinement(nn.Module):
    def __init__(self, channels):
        super(Refinement, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample = nn.AvgPool2d(2)

        self.E0 = EncoderB(1, channels[0] + 1, channels[0], downsample=False)   # scale: 1/4
        self.E1 = EncoderB(1, channels[0],     channels[1], downsample=True)    # scale: 1/8
        self.E2 = EncoderB(1, channels[1],     channels[2], downsample=True)    # scale: 1/16
        self.E3 = EncoderB(1, channels[2],     channels[3], downsample=True)    # scale: 1/32

        self.D0 = EncoderB(1, channels[4],     channels[4], downsample=False)   # scale: 1/32
        self.D1 = DecoderB(1, channels[4] + channels[5], channels[5])           # scale: 1/16
        self.D2 = DecoderB(1, channels[5] + channels[6], channels[6])           # scale: 1/8
        self.D3 = DecoderB(1, channels[6] + channels[7], channels[7])           # scale: 1/4
        self.D4 = DecoderB(1, channels[7],               channels[8])           # scale: 1/2
        self.D5 = DecoderB(1, channels[8],               channels[9])           # scale: 1

        # regression
        self.confidence = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.disp = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False)
        )

    def forward(self, fea, disp):
        # scale the input disparity
        disp = disp / (2 ** 5)
        # print("====", "disp_ori:", disp.shape)

        fea_E0 = self.E0(torch.cat((disp, fea), 1))                         # scale: 1/4
        fea_E1 = self.E1(fea_E0)                                            # scale: 1/8
        fea_E2 = self.E2(fea_E1)                                            # scale: 1/16
        fea_E3 = self.E3(fea_E2)                                            # scale: 1/32

        fea_D0 = self.D0(fea_E3)                                            # scale: 1/32
        fea_D1 = self.D1(torch.cat((self.upsample(fea_D0), fea_E2), 1))     # scale: 1/16
        fea_D2 = self.D2(torch.cat((self.upsample(fea_D1), fea_E1), 1))     # scale: 1/8
        fea_D3 = self.D3(torch.cat((self.upsample(fea_D2), fea_E0), 1))     # scale: 1/4
        # print("===shapeoffeature3:",fea_D3.shape)
        fea_D4 = self.D4(self.upsample(fea_D3))                             # scale: 1/2
        fea_D5 = self.D5(self.upsample(fea_D4))                             # scale: 1
        # print("===shapeoffeature5:",fea_D5.shape)
        # regression
        confidence = self.confidence(fea_D5)
        disp_res = self.disp(fea_D5)
        disp_res = torch.clamp(disp_res, 0)

        disp = F.interpolate(disp, scale_factor=4, mode='bilinear') * (1-confidence) + disp_res * confidence
        # print("===","disp_after_before2+7:",disp.shape)
        # scale the output disparity
        # note that, the size of output disparity is 4 times larger than the input disparity
        result_disp =disp * 2 ** 7
        # print("===","disp_after:",result_disp.shape)
        return disp * 2 ** 7


class DepthEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(DepthEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),

                                  )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),
                                  )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x2=None, pre_x3=None, pre_x4=None):
        ### input

        x0 = self.init(input)
        if pre_x4 is not None:
            x0 = x0 + F.interpolate(pre_x4, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        if pre_x3 is not None:  # newly added skip connection
            x1 = x1 + F.interpolate(pre_x3, scale_factor=scale, mode='bilinear', align_corners=True)

        x2 = self.enc2(x1)  # 1/4 input size
        if pre_x2 is not None:  # newly added skip connection
            x2 = x2 + F.interpolate(pre_x2, scale_factor=scale, mode='bilinear', align_corners=True)

        return x0, x1, x2


class RGBEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(RGBEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc3 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc4 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x=None):
        ### input

        x0 = self.init(input)
        if pre_x is not None:
            x0 = x0 + F.interpolate(pre_x, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        x2 = self.enc2(x1)  # 1/4 input size
        x3 = self.enc3(x2)  # 1/8 input size
        x4 = self.enc4(x3)  # 1/16 input size

        return x0, x1, x2, x3, x4


class DepthDecoder(nn.Module):
    def __init__(self, layers, filter_size):
        super(DepthDecoder, self).__init__()
        padding = int((filter_size - 1) / 2)

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                  )

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                  )

        self.prdct = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                   nn.ReLU(),
                                   
                                   nn.Conv2d(layers // 2, 1, filter_size, stride=1, padding=padding))

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, pre_dx, pre_cx):
        # print("========", pre_cx[2].shape, pre_dx[2].shape)
        x2 = pre_dx[2] + pre_cx[2]  # torch.cat((pre_dx[2], pre_cx[2]), 1)
        x1 = pre_dx[1] + pre_cx[1]  # torch.cat((pre_dx[1], pre_cx[1]), 1) #
        x0 = pre_dx[0] + pre_cx[0]

        x3 = self.dec2(x2)  # 1/2 input size
        x4 = self.dec1(x1 + x3)  # 1/1 input size

        ### prediction
        output_d = self.prdct(x4 + x0)

        return x2, x3, x4, output_d


class Refinement_new(nn.Module):
    def __init__(self):
        super(Refinement_new, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        denc_layers = 32 # 32
        cenc_layers = 32 # 32
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder(ddcd_layers, 3)

    def forward(self, input_d, input_rgb):
        # print("====", input_d.shape, input_rgb.shape)
        
        input_d_half = self.upsample(input_d)                             # scale: 1/2
        input_d = self.upsample(input_d_half)
        # print("====after===",input_d.shape)
        C = (input_d > 0).float()

        
        enc_c = self.rgb_encoder(input_rgb)

        
        ## for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        ## for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        ## for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11
        output_d12 = predict_d11
        output_d14 = F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)
        # print("====result====",output_d11.shape)
        return output_d11 #, output_d12, output_d14







