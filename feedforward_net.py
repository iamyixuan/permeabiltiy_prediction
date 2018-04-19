import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# hyperparameters


batch_size = 50
learning_rate = 0.0001
num_epoch = 100

file_label = pd.ExcelFile("permeability.xlsx")
labels = file_label.parse(header = None)
data_indx = np.load('index.npy','r')

labels_sub = labels.iloc[list(data_indx)]

data = np.load('extracted_feat.npy','r')

def shuffle_split(data, label):
	#np.random.seed(10)
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

train_loader = DataLoader(dataset = training,
						  batch_size= batch_size,
						  shuffle = False)
test_loader = DataLoader(dataset = testing,
						 batch_size = batch_size,
						 shuffle = False)
val_loader = DataLoader(dataset = validation,
						batch_size = batch_size,
						shuffle = False)



class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Sequential(
			nn.Linear(100, 150),
			nn.ReLU(),
			nn.Linear(150, 80),
			nn.ReLU(),
			nn.Linear(80, 50),
			nn.ReLU(),
			nn.Linear(50,1))

	def feedforward(self,x):
		out = self.fc1(x)
		return out

net = Net()

# loss function
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

# train the model
train_err = []
val_err = []

for epoch in range(num_epoch):
	print epoch
	optimizer.zero_grad()
	for dat, tar in train_loader: 
		structure = Variable(dat.view(-1, 100))
		permeability = Variable(tar.view(-1,1))
		#forward+backprop
		
		outputs = net.feedforward(structure)
		loss = criterion(outputs, permeability)
		loss.backward()
	optimizer.step()

	for val_d, val_t in val_loader:
		v_structure = Variable(val_d.view(-1, 100))
		v_label = Variable(val_t.view(-1,1))

		out_v = net.feedforward(v_structure)
		v_loss = criterion(out_v, v_label)


	print('Epoch %s'% epoch, 'Train_Loss %s'% loss.data[0],
		'Val_loss %s' % v_loss.data[0])
	train_err.append(loss.data[0])
	val_err.append(v_loss.data[0])

torch.save(net.state_dict(), 'feedforward.pkl')
#print train_err, val_err
# test the model
net.eval()
test_err = []
for test_dat, test_tar in test_loader:
	t_structure = Variable(test_dat.view(-1, 100))
	label = Variable(test_tar.view(-1,1))

	out = net.feedforward(t_structure)
	diff = label - out
	mae = np.mean(np.absolute(diff.data.numpy()))

	test_err.append(mae)

	print('Test Error %s' % np.mean(np.array(test_err)))

plt.plot(range(num_epoch), train_err, color = 'blue', linewidth = 3, label = 'Training Error')
plt.plot(range(num_epoch), val_err,  color = 'red', linewidth = 1.5, label = 'Validation Error')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
plt.ylabel('MSE')
plt.show()



