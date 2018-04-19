import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
import gc

#hyper parameters
num_epoch = 50
batch_size = 30
learning_rate = 0.001


#load the data

f = open('subset_rand.json','rb')
gc.disable()
data = json.load(f)
gc.enable()
f.close

#load the label

file_label = pd.ExcelFile("permeability.xlsx")
labels = file_label.parse(header = None)
data_indx = np.load('index.npy','r')
labels_sub = labels.iloc[data_indx]

# shuffle and split data into train and test sets

def shuffle_split(data, label):
	data = np.array(data)
	label = np.array(label)
	label = 10**11 * label
	if len(data) == len(label):
		print 'checked out'
	indx = np.random.permutation(len(data))
	test_size = int(0.2*len(data))
	test_indx = indx[:test_size]
	train_indx = indx[test_size:]
	val_indx = train_indx[:test_size]
	train_indx = train_indx[test_size:]
	train_dat, train_tar, val_dat, val_tar, test_dat, test_tar = data[train_indx], label[train_indx], data[val_indx], label[val_indx], data[test_indx], label[test_indx]
	return train_dat, train_tar, val_dat, val_tar, test_dat, test_tar

train_dat, train_tar, val_dat, val_tar, test_dat, test_tar = shuffle_split(data, labels_sub)

del(data, file_label, labels_sub) # free memory
#instantiate a dataset
class Pore_data(Dataset):
	def __init__(self, X, y, transform = None):
		self.X = X
		self.y = y
		self.transform = transform
	def __len__(self):
		return len(self.X)
	def __getitem__(self, idx):
		X = torch.from_numpy(self.X[idx]).float()
		y = torch.from_numpy(self.y[idx]).float()
		return X, y


training = Pore_data(train_dat, train_tar)
testing = Pore_data(test_dat, test_tar)
validation = Pore_data(val_dat, val_tar)

del(train_dat, train_tar, test_dat, test_tar) #free memory

# load the data in batches and right format.

train_loader = DataLoader(dataset = training,
						  batch_size= batch_size,
						  shuffle = False)
test_loader = DataLoader(dataset = testing,
						 batch_size = batch_size,
						 shuffle = False)
val_loader = DataLoader(dataset = validation,
						batch_size = batch_size,
						shuffle = False)







#convnet
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(100, 200, kernel_size = 5, padding = 2),
			nn.ReLU(),
			nn.Conv2d(200, 150, kernel_size = 5, padding = 2),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(150, 50, kernel_size = 5, padding = 2),
			nn.ReLU(),
			nn.Conv2d(50, 14, kernel_size = 5, padding = 2),
			nn.ReLU()) # 400*4
		self.layer3 = nn.Sequential(
			nn.Conv2d(14, 28, kernel_size = 5, padding = 2),
			nn.ReLU(),
			nn.Conv2d(28, 28, kernel_size = 5, padding = 2),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer4 = nn.Sequential(
			nn.Conv2d(28, 56, kernel_size = 5, padding = 2),
			nn.ReLU(),
			nn.MaxPool2d(5))

		self.fc1 = nn.Sequential(
			 nn.Linear(5*5*56, 500),
			 nn.ReLU())

		self.fc2 = nn.Linear(500, 1)


	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.view(out.size(0), -1)
		out = self.fc1(out)
		out = self.fc2(out)
		return out

cnn = CNN()
#loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters())

#Train the model
train_err = []
val_err = []

for epoch in range(num_epoch):
	print epoch
	optimizer.zero_grad()
	for dat, tar in train_loader: 
		structure = Variable(dat.view(-1,100,100,100))
		permeability = Variable(tar.view(-1,1))
		#forward+backprop
		
		outputs = cnn.forward(structure)
		loss = criterion(outputs, permeability)
		loss.backward()
	optimizer.step()

	for val_d, val_t in val_loader:
		v_structure = Variable(val_d.view(-1, 100, 100, 100))
		v_label = Variable(val_t.view(-1,1))

		out_v = cnn.forward(v_structure)
		v_loss = criterion(out_v, v_label)


	print('Epoch %s'% epoch, 'Train_Loss %s'% loss.data[0],
		'Val_loss %s' % v_loss.data[0])
	train_err.append(loss.data[0])
	val_err.append(v_loss.data[0])
print train_err, val_err
# test the model
cnn.eval()
test_err = []
for test_dat, test_tar in test_loader:
	t_structure = Variable(test_dat.view(-1, 100, 100, 100))
	label = Variable(test_tar.view(-1,1))

	out = cnn.forward(t_structure)
	diff = label - out
	mae = np.mean(np.absolute(diff.data.numpy()))

	test_err.append(mae)

	print('Test Error %s' % np.mean(np.array(test_err)))










