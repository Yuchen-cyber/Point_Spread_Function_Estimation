"""
The code for generating the psf ground truth images are modified from https://github.com/Lukeli0425/LR-PSF.
"""
import csv
import torch
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import os
import torch.nn.functional as F
from torch.fft import fftn, ifftn, fftshift, ifftshift
# reading the data
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print("###########################")
print('Reading Data')
file_path = '/home/yuchenwang/outputs/parameters.csv' 
labels= []
psfs = []
test_number = 0
with open(file_path, newline='') as psf_data:
    if test_number <= 5000 :
        test_number += 1
        reader = csv.reader(psf_data)
        for data in reader:
            id = data[0]
            data = data[1:]
            converted_data = [float(s) for s in data]
            labels.append(converted_data)
            psf_file_path = '/mnt/WD6TB/yuchen_wang/dataset/LSST_23.5_EM/psf/psf_' + id + '.pth'
            psf = torch.load(psf_file_path)
            psfs.append(psf)


# model
class CNN_MLP(nn.Module):
    def __init__(self):
        super(CNN_MLP, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.batchNor = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchNor2 = nn.BatchNorm2d(64)

        self.fc1_transition = nn.Linear(9216, 64)
        # self.fc1_transition = nn.Linear((32/4) * (64/4) * 64, 64)

        # MLP layers
        nn.Dropout()
        self.fc1_mlp = nn.Linear(64, 128)
        self.fc2_mlp = nn.Linear(128, 64)
        self.fc3_mlp = nn.Linear(64, 3)

    def forward(self, x):
        # CNN
        x = self.pool(F.relu(self.batchNor(self.conv1(x))))
        x = self.pool(F.relu(self.batchNor2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_transition(x))

        # MLP
        x = F.relu(self.fc1_mlp(x))
        nn.Dropout()
        x = F.relu(self.fc2_mlp(x))
        x = self.fc3_mlp(x)

        return x



print("###########################")
print('Building models')
model = CNN_MLP().to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("###########################")
print('process data')
#process data
psfs = torch.stack(psfs)
labels =np.stack(labels)
print("###########################")
print('Nomalization')
atoms_fwhm_mean = np.mean(labels[:,0])
atmos_fwhm_std = np.std(labels[:,0])
atoms_e_mean = np.mean(labels[:,1])
atmos_e_std = np.std(labels[:,1])
atoms_beta_mean = np.mean(labels[:,2])
atmos_beta_std = np.std(labels[:,2])
mean_list = np.array([atoms_fwhm_mean, atoms_e_mean, atoms_beta_mean])
std_list = np.array([atmos_fwhm_std, atmos_e_std, atmos_beta_std])
labels = (labels - mean_list) / std_list

labels = torch.tensor(labels, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(psfs.numpy(), labels.numpy(), test_size=0.2)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)


#train
print("###########################")
print('Training')
num_epochs = 30
nn_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    max_loss = 1
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss/len(train_loader.dataset)
    nn_losses.append(epoch_loss) 
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")
    
print("###########################")
print('Testing')
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        target_param = targets.cpu()[0] * std_list + mean_list
        input = inputs[0]
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
    
    
    epoch_loss = test_loss/len(test_loader.dataset)
    print(f"Test Loss: {test_loss/len(test_loader.dataset):.4f}")

   
# normalise the input
errors = abs((outputs - targets)).to('cuda')
errors_non_nomalised = errors.cpu().numpy() * std_list
mean_errors = torch.mean(errors, axis=0)
std_errors = torch.std(errors, axis=0)

"""
The reason why draw two loss graphs is because the ranges are different for different parameters.
"""
# save the image for atmos_beta: ranging from 0 to 2pi
mean_errors = torch.mean(torch.tensor(errors_non_nomalised), axis=0)[2]
std_errors = torch.std(torch.tensor(errors_non_nomalised), axis=0)[2]
parameters = ['atmos_beta']
fig, ax = plt.subplots()
bars = ax.bar(parameters, mean_errors.cpu().numpy(), yerr=std_errors.cpu().numpy(), align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Error')
ax.set_title('Mean Error with Error Bars for Each Parameter')
ax.yaxis.grid(True)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 6), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('/home/yuchenwang/outputs/images/loss_plot_atmos_beta.png', format='png')

# save the image for other parameters
mean_errors = torch.mean(torch.tensor(errors_non_nomalised), axis=0)[0:2]
std_errors = torch.std(torch.tensor(errors_non_nomalised), axis=0)[0:2]
parameters = ['atomos_fwhm', 'atmos_e']
fig, ax = plt.subplots()
bars = ax.bar(parameters, mean_errors.cpu().numpy(), yerr=std_errors.cpu().numpy(), align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Error')
ax.set_title('Mean Error with Error Bars for Each Parameter')
ax.yaxis.grid(True)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval*0.0001, round(yval, 6), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('/home/yuchenwang/outputs/images/loss_plot.png', format='png')
