import numpy as np
import torch
import cv2
from tqdm import tqdm
import torch.nn as nn
from model.sebnet import SEBNet
from utils.dataset import FundusSeg_Loader
import copy
import time
from sklearn.metrics import roc_auc_score
from fvcore.nn import FlopCountAnalysis, parameter_count_table

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

dataset_name="idrid"
model_path='./snapshot/idrid_pretrain.pth'

if dataset_name == "idrid":
    test_data_path = "./data/idrid/test/"
    raw_height = 712
    raw_width =  1072

if dataset_name == "ddr":
    test_data_path = "xxx/ddr/test/"
    raw_height = 712
    raw_width =  1072

if dataset_name == "fgadr":
    test_data_path = "xxx/fgadr/train/"#CV2
    raw_height = 512
    raw_width =  512

save_path='./results/'

def save_results(pred, save_path, filename):
    pred_ex = pred.cpu().numpy().astype(np.double)[0][1] 
    pred_he = pred.cpu().numpy().astype(np.double)[0][2] 
    pred_ma = pred.cpu().numpy().astype(np.double)[0][3] 
    pred_se = pred.cpu().numpy().astype(np.double)[0][4] 

    pred_all = pred.cpu().argmax(dim=1)[0] # 0, 1, 2, 3, 4
    pred_all = pred_all.numpy().astype(np.double)

    pred_ex = pred_ex * 255
    pred_he = pred_he * 255
    pred_ma = pred_ma * 255
    pred_se = pred_se * 255
    
    # Background-> Black: 0 0 0
    # EX->Red:    255 0   0
    # HE->Green:  0   255 0
    # MA->White:  255 255 255 
    # SE->Yellow: 255 255 0

    c1=np.zeros((pred_all.shape[0], pred_all.shape[1]),dtype="uint8")
    c2=np.zeros((pred_all.shape[0], pred_all.shape[1]),dtype="uint8")
    c3=np.zeros((pred_all.shape[0], pred_all.shape[1]),dtype="uint8")
    c1[pred_all==1]=0
    c2[pred_all==1]=0
    c3[pred_all==1]=255
    c1[pred_all==2]=0
    c2[pred_all==2]=255
    c3[pred_all==2]=0
    c1[pred_all==3]=255
    c2[pred_all==3]=255
    c3[pred_all==3]=255
    c1[pred_all==4]=0
    c2[pred_all==4]=255
    c3[pred_all==4]=255
    pred_color=np.stack([c1,c2,c3],axis=-1)

    cv2.imwrite(save_path + filename[0] + '_ex.png', pred_ex)
    cv2.imwrite(save_path + filename[0] + '_he.png', pred_he)
    cv2.imwrite(save_path + filename[0] + '_ma.png', pred_ma)
    cv2.imwrite(save_path + filename[0] + '_se.png', pred_se)
    cv2.imwrite(save_path + filename[0] + '_all.png', pred_color)

if __name__ == "__main__":
    with torch.no_grad():
        test_dataset = FundusSeg_Loader(test_data_path,0, dataset_name)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        print('Testing images: %s' %len(test_loader))
        device = torch.device('cuda')
        net = SEBNet(n_channels=3, n_classes=5)

        #### Bench Model ##
        #tensor = (torch.rand(1, 3, 512, 512),)
        #flops = FlopCountAnalysis(net, tensor)
        ##print("FLOPs: ", flops.total())
        #print(parameter_count_table(net))

        criterion = nn.CrossEntropyLoss()
        net.to(device=device)
        print(f'Loading model {model_path}')
        net.load_state_dict(torch.load(model_path, map_location=device))

        ###

        net.eval()
        total_loss = 0
        total_time = 0
        for image, label, filename in test_loader:
            image = image.cuda().float()
            label = label.cuda().float()

            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            torch.cuda.synchronize()
            start = time.time()
            pred = net(image)
            torch.cuda.synchronize()
            end = time.time()
            total_time = total_time + (end-start)
            #print('Inference time: %s seconds'%(end-start))

            loss = criterion(pred, label.long())
            total_loss = total_loss + loss.item()
            pred = torch.softmax(pred, dim=1)
            # Save Segmentation Maps
            save_results(pred, save_path, filename) 

        print('Loss/Test', total_loss)

