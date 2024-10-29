from model.sebnet import SEBNet
from utils.dataset import FundusSeg_Loader
from torch import optim
import torch.nn as nn
import torch
import sys
from tqdm import tqdm
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


dataset_name = "idrid" # 

if dataset_name == "idrid":
    train_data_path = "xxx/idrid/train/"
    valid_data_path = "xxx/idrid/test/"
    N_epochs = 3000
    lr_decay_step = [1500, 2800]
    lr_init = 0.001
    batch_size = 8
    test_epoch = 10

def train_net(net, device, epochs=N_epochs, batch_size=batch_size, lr=lr_init):
    train_dataset = FundusSeg_Loader(train_data_path, 1, dataset_name)
    valid_dataset = FundusSeg_Loader(valid_data_path, 0, dataset_name)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    print('Traing images: %s' % len(train_loader))
    print('Valid  images: %s' % len(valid_loader))

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=lr_decay_step,gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    train_loss_list = []
    val_loss_list = []
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        net.train()
        train_loss = 0
        for i, (image, label, filename) in enumerate(train_loader):
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
            loss = criterion(pred, label.long())
            train_loss = train_loss + loss.item()
            loss.backward()
            optimizer.step()

        train_loss_list.append(train_loss / i)

        # Validation
        if ((epoch+1) % test_epoch == 0):
            net.eval()
            val_loss = 0
            for i, (image, label, filename) in enumerate(valid_loader):
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                loss = criterion(pred, label.long())
                val_loss = val_loss + loss.item()
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(net.state_dict(), './snapshot/idrid.pth')
                print('saving model............................................')
        
            val_loss_list.append(val_loss / i)
            print('Loss/valid', val_loss / i)
            sys.stdout.flush()

        scheduler.step()

if __name__ == "__main__":
    device = torch.device('cuda')
    net = SEBNet(n_channels=3, n_classes=5)
    net.to(device=device)
    train_net(net, device)
