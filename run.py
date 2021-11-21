import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import itertools
import cv2 as cv
import os
import numpy as np
from PIL import Image
import argparse




def tensor_to_png(tensor, filename):
    tensor = tensor.view(tensor.shape[1:])
    if use_cuda:
        tensor = tensor.cpu()
    tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    pil = tensor_to_pil(tensor)
    pil.save(filename)

def png_to_tensor(ground_truth_path):
    pil = Image.open(ground_truth_path).convert('RGB')
    #pil = cv.resize(pil, (256,256), interpolation=cv.INTER_CUBIC)
    pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    if use_cuda:
        tensor = pil_to_tensor(pil).cuda()
    else:
        tensor = pil_to_tensor(pil)
    return tensor.view([1]+list(tensor.shape))



def pngbw_to_tensor(ground_truth_path):
    pil = Image.open(ground_truth_path).convert('RGB')
    #pil = cv.resize(pil, (256,256), interpolation=cv.INTER_CUBIC)
    pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    if use_cuda:
        tensor = pil_to_tensor(pil).cuda()
    else:
        tensor = pil_to_tensor(pil)
    tensor = tensor[0,:,:]
    tensor = tensor.view([1,1]+list(tensor.shape))

    return tensor



class residual_block(nn.Module):
    def __init__(self):
        super(residual_block, self).__init__()
        self.conv_1 = nn.Conv2d(48, 48, 3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(48)

        self.conv_2 = nn.Conv2d(48, 48, 3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(48)
 
    def forward(self, input):
        output = self.conv_1(input)
        output = self.bn_1(output)
        output = F.relu(output)

        output = self.conv_2(output)
        output = self.bn_2(output)
        
        return output + input


class DCR(nn.Module):
    def __init__(self):
        super(DCR, self).__init__()
        self.conv_1 = nn.Conv2d(7, 64, 3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)

        self.block2 = residual_block()
        self.block3 = residual_block()

        self.conv_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(64)

        self.conv_4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn_4 = nn.BatchNorm2d(64)
        self.conv_5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn_5 = nn.BatchNorm2d(64)

        self.block6 = residual_block()
        self.block7 = residual_block()

        self.conv_8 = nn.Conv2d(64 ,3, 3, stride=1, padding=1)

        self.conv_9 = nn.Conv2d(3, 48, 3, stride=1, padding=1)
        self.bn_9 = nn.BatchNorm2d(48)

        self.block10 = residual_block()
        self.block11 = residual_block()

        self.conv_12 = nn.Conv2d(48, 48, 3, stride=1, padding=1)

        self.conv_13 = nn.Conv2d(48, 48, 3, stride=1, padding=1)
        self.bn_13 = nn.BatchNorm2d(48)

        self.block14 = residual_block()
        self.block15 = residual_block()

        self.conv_16 = nn.Conv2d(48, 3, 3, stride=1, padding=1)

        
    def forward(self, O1,S1VH,S1VV,S2VH,S2VV):
        inputim =  torch.cat((O1,S1VH,S1VV,S2VH,S2VV),1)
        output = self.conv_1(inputim)
        output = self.bn_1(output)
        output = F.relu(output)

        output = self.conv_2(output)
        output = self.bn_2(output)
        output = F.relu(output)

        output = self.conv_3(output)
        output = self.bn_3(output)
        output = F.relu(output)

        output = self.conv_4(output)
        output = self.bn_4(output)
        output = F.relu(output)
        #output = F.leaky_relu(output)
        #output = self.block2(output)
        #output = self.block3(output)


        output = self.conv_5(output)
        output = self.bn_5(output)
        output = F.leaky_relu(output)
        #output = self.block6(output)
        #output = self.block7(output)
        output = self.conv_8(output)
        output = nn.Sigmoid()(output)

        #output = self.conv_9(recons)
        #output = self.bn_9(output)
        #output = F.leaky_relu(output)
        #output = self.block10(output)
        #output = self.block11(output)
        #output = self.conv_12(output)
        #alpha2 = F.relu(output)

        #output = self.conv_13(alpha2)
        #output = self.bn_13(output)
        #output = F.leaky_relu(output)
        #output = self.block14(output)
        #output = self.block15(output)
        #recyc = self.conv_16(output)

        return output

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indexes', type=int,default = 1)
    args = parser.parse_args()
    indexes = args.indexes
    use_cuda = True
    use_cuda = True
    S1VH_path = './' + str(indexes) + '/S1VH.png'
    S1VV_path = './' + str(indexes) + '/S1VV.png'
    S2VH_path = './' + str(indexes) + '/S2VH.png'
    S2VV_path = './' + str(indexes) + '/S2VV.png'
    O1_path = './' + str(indexes) + '/O1.png'
    O2_path = './' + str(indexes) + '/O2.png'
    opt_path = './' + str(indexes) + '/optcrop.png'
    mask_path = './' + str(indexes) + '/maskcrop.png'

    S1VH = pngbw_to_tensor(S1VH_path)
    S1VV = pngbw_to_tensor(S1VV_path)
    S2VH = pngbw_to_tensor(S2VH_path)
    S2VV = pngbw_to_tensor(S2VV_path)
    O1 = png_to_tensor(O1_path)
    O2 = png_to_tensor(O2_path)
    opt = png_to_tensor(opt_path)
    mask = png_to_tensor(mask_path)
    mask = 1 - mask
    res_opt0 = O2 * mask + opt
    res_opt = O2 * mask

    recons_path = './' + str(indexes) + '/output/'
    mkdir(recons_path)

    num_steps = 1001
    save_frequency = 100
    net = DCR()
    if use_cuda:
        net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
    save_img_ind = 0

    for step in range(num_steps):
        loss_main1 = 0
        loss_main2 = 0
        for w in range(10):
            for h in range(10):
                O1crop = O1[:,:,w*200:(w+1)*200,h*200:(h+1)*200]
                O2crop = O2[:,:,w*200:(w+1)*200,h*200:(h+1)*200]
                S1VHcrop = S1VH[:,:,w*200:(w+1)*200,h*200:(h+1)*200]
                S1VVcrop = S1VV[:,:,w*200:(w+1)*200,h*200:(h+1)*200]
                S2VHcrop = S2VH[:,:,w*200:(w+1)*200,h*200:(h+1)*200]
                S2VVcrop = S2VV[:,:,w*200:(w+1)*200,h*200:(h+1)*200]
                maskcrop = mask[:,:,w*200:(w+1)*200,h*200:(h+1)*200]
                res_optcrop = res_opt[:,:,w*200:(w+1)*200,h*200:(h+1)*200]
                res_opt0crop = res_opt0[:,:,w*200:(w+1)*200,h*200:(h+1)*200]

                output = net(O1crop,S1VHcrop,S1VVcrop,S2VHcrop,S2VVcrop)
                mask_output = output * maskcrop

                optimizer.zero_grad()

                loss_total = torch.sum(torch.abs(mask_output - res_optcrop))/40000
                loss_real = torch.sum(torch.abs(output - O2crop))/40000
                loss_main1 = loss_main1 + loss_real
                loss_main2 = loss_main2 + 10*torch.log10(1/torch.nn.MSELoss()(output, O2crop))
                loss_total.backward()
                optimizer.step()
                if step % save_frequency == 0:
                    tensor_to_png(output.data,recons_path+'{}_{}_{}.png'.format(save_img_ind, w, h))

        
        print('At step {}, loss_main1 is {}, loss_main2 is {}'.format(step, loss_main1.data.cpu(), loss_main2.data.cpu()/100))
        
        if step % save_frequency == 0:
            save_img_ind += 1
    if use_cuda:
        torch.cuda.empty_cache()



