import random, os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#from torchvision.models import resnet18
from model import resnet18

def same_seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


# train and validate
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, nums_epochs):
    train_losses = []
    valid_losses = []
    #best_valid_loss = np.inf
    best_valid_acc = -10
    for epoch in range(nums_epochs):
         # model train
        model.train()
        train_loss = 0
        for data, labels in tqdm(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        scheduler.step()
        
        #model validate
        model.eval()
        valid_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                
                loss = criterion(output, labels)
                valid_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
            
            
            epoch_acc = accuracy_score(y_true, y_pred)
            if epoch_acc > best_valid_acc:
                best_valid_acc = epoch_acc
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }            
                torch.save(state, f'./checkpoint/best_loss.pth')   
            valid_loss /= len(test_loader)
            valid_losses.append(valid_loss)
            print(f'[Epoch {epoch}] Train loss: {train_loss} Valid loss: {valid_loss} Acc:{epoch_acc}') 
            
    # inference
    checkpoint = torch.load(f'./checkpoint/best_loss.pth')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1, keepdim=True)
            
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(pred.cpu().numpy())
            
    test_acc = accuracy_score(y_true_test, y_pred_test)
    print(f'Best Acc:{test_acc}')
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.text(len(train_losses)-1, train_losses[-1], f'Best Accuracy: {test_acc}%', horizontalalignment='right')
    plt.savefig('./plot/loss.png')
 
if __name__ == '__main__':
    same_seed_everywhere(1011)
    transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=100, dropout_prob=0.35) 
    #model.fc = nn.Linear(model.fc.in_features, 100) 
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.01, epochs = 50, 
                                                steps_per_epoch=len(train_loader))
    
    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, nums_epochs=50)
    