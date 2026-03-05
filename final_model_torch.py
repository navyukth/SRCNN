import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import torch.nn.init as init

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def downsample_image(image, scale_factor=4):
    blurred = cv2.GaussianBlur(image, (31,31), sigmaX=4)
    h, w = image.shape[:2]
    low_res = cv2.resize(blurred, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_AREA)
    return low_res

class ImageDataset(Dataset):
    def __init__(self, high_res_path, file_list, scale_factor=4):
        self.file_list = file_list
        self.high_res_path = high_res_path
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        high_res_img = cv2.imread(os.path.join(self.high_res_path, file_name))
        high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_BGR2RGB)
        l=high_res_img.shape[0]
        b=high_res_img.shape[1]
        high_res_img = high_res_img[(l//2)-200:(l//2)+200,(b//2)-200:(b//2)+200,:]
        high_res_img = high_res_img.astype('float32') / 255.0

        low_res_img = downsample_image(high_res_img, scale_factor=self.scale_factor)

        return (
            torch.tensor(high_res_img.transpose(2, 0, 1)),
            torch.tensor(low_res_img.transpose(2, 0, 1))
        )

train_path = 'dataset/DIV2K_train_HR/DIV2K_train_HR'
test_path = 'dataset/DIV2K_test_HR/DIV2K_test_HR'

train_files = os.listdir(train_path)
test_files = os.listdir(test_path)

train_dataset = ImageDataset(train_path, train_files, scale_factor=4)
test_dataset = ImageDataset(test_path, test_files, scale_factor=4)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the SRCNN model
class SRCNN(nn.Module):
    def __init__(self, scale_factor=4):
        super(SRCNN, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)
        self.conv1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.leaky_relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.leaky_relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x=self.upsample(x)
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        x = self.leaky_relu3(self.conv3(x))
        x = self.conv4(x)
        return x
    
def init_weights_he(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    
def PSNR(y_true, y_pred):
    mse=criterion(y_true,y_pred)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

# Initialize the model
model = SRCNN().cuda()
model.apply(init_weights_he)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss=[]
train_psnr=[]

# Training the model
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    epoch_psnr = 0
    epoch_loss = 0
    for high_res, low_res in tqdm(train_loader):
        high_res, low_res = high_res.cuda(), low_res.cuda()
        optimizer.zero_grad()
        outputs = model(low_res)
        loss = criterion(outputs, high_res)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_psnr += PSNR(high_res, outputs).item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    train_loss.append(epoch_loss)
    train_psnr.append(epoch_psnr/len(train_loader))

epochs=[i for i in range(1,num_epochs+1)]

# plt.plot(epochs,train_loss,marker='o',linestyle='-',color='b')
# plt.title("MSE Loss curve")
# plt.xlabel("Epochs")
# plt.ylabel("MSE Loss")
# plt.show()

# plt.plot(epochs,train_psnr,marker='o',linestyle='-',color='b')
# plt.title("PSNR curve")
# plt.xlabel("Epochs")
# plt.ylabel("PSNR")
# plt.show()

# Function to plot images
def plot_images(high, low, predicted):
    high = high.permute(1, 2, 0).cpu().numpy()
    low = low.permute(1, 2, 0).cpu().numpy()
    predicted = predicted.permute(1, 2, 0).detach().cpu().numpy()

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.title('High Image', color='green', fontsize=20)
    plt.imshow(high)
    plt.subplot(1, 3, 2)
    plt.title('Low Image', color='black', fontsize=20)
    plt.imshow(low)
    plt.subplot(1, 3, 3)
    plt.title('Predicted Image', color='red', fontsize=20)
    plt.imshow(predicted)
    plt.show()

torch.save(model.state_dict(),'model_weights.pth') #saved for 200 epochs

# Testing the model
model.eval()

test_psnr=0

for i, (high_res, low_res) in enumerate(test_loader):
    high_res, low_res = high_res.cuda(), low_res.cuda()
    predicted = model(low_res)
    psnr=PSNR(high_res[0], predicted[0]).item()
    test_psnr+=psnr
print("Average test PSNR: ",test_psnr/len(test_dataset))

show_high=[]
show_low=[]
show_list=[3,2,1,0]

for i in show_list:
    show_high.append(test_files[i])

show_loader = DataLoader(ImageDataset(test_path, show_high, scale_factor=4), batch_size=1)

for i,(high_res,low_res) in enumerate(show_loader):
    high_res, low_res = high_res.cuda(), low_res.cuda()
    predicted = model(low_res)
    plot_images(high_res[0], low_res[0], predicted[0])
    psnr = PSNR(high_res[0], predicted[0]).item()
    print('PSNR:', psnr, 'dB')
    











