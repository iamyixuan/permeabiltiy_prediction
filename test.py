
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
import gc
import time
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch import nn, optim
from torch.autograd import Variable
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader


input_size = 3
hidden_size1 = 10
hidden_size2 = 8
hidden_size3 = 5
num_epoch = 350
learning_rate = 0.001
batch_size = 5
start = time.time()
'''
------------------
generating the data:
Features: void_ratio:e ; porosity; e^3/1+e;
Target: permeability

combining them together into a (1000,4) array
The last column is the target value
-------------------
'''


f = open('void_ratio.json','rb')
gc.disable()
data = json.load(f)
gc.enable()
f.close
data = np.array(data)
data_1 = 1.0/data # e void_ratio


file_label = pd.ExcelFile("permeability.xlsx")
labels = file_label.parse(header = None)
data_indx = np.load('index.npy','r')
labels_sub = labels.iloc[data_indx] # permeability: out targets

void_ratio = []
para = []
for i in range(1000):
	e = data_1[i]
	par = e**3.0/(1+e)
	void_ratio.append(e)
	para.append(par)  # para: e^3/1+e---one of the features. 

void_ratio = np.array(void_ratio)
mid = pd.ExcelFile("porosity.xlsx")
poro = mid.parse(header = None)
porosity = poro.iloc[data_indx]
porosity = np.array(porosity).reshape(1000,) # porosity: one of the features.

data = np.vstack((void_ratio, porosity))
data = np.vstack((data, para)).T
print data.shape
#data = np.vstack((data, labels_sub)).T
# put it into a data frame

#data_df = pd.DataFrame(data, columns = ['void_ratio', 'porosity','e^3/1+e', 'permeability'])

'''
-------------------
shuffle the data 
split it into training set and testing set
-------------------
'''
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

'''
--------------
building up NN for training
--------------
'''


class Nerual_net(nn.Module):
	def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
		super(Nerual_net,self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size1)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size1, hidden_size2)
		self.relu2 = nn.ReLU()
		self.fc3 = nn.Linear(hidden_size2,hidden_size3)
		self.relu3 = nn.ReLU()
		self.fc4 = nn.Linear(hidden_size3,5)
		self.relu4 = nn.ReLU()
		self.fc5 = nn.Linear(5, 1)

	def feed_forward(self, x):
		out = self.fc1(x)
		out = self.relu1(out)
		out = self.fc2(out)
		out = self.relu2(out)
		out = self.fc3(out)
		out = self.relu3(out)
		out = self.fc4(out)
		out = self.relu4(out)
		out = self.fc5(out)
		return out

net = Nerual_net(input_size, hidden_size1, hidden_size2, hidden_size3)


criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

# training the data combining 10-fold validation

# kf = KFold(n_splits = 10)
# kf.get_n_splits(train_set)







train_err = []
val_err = []

for epoch in range(num_epoch):
	#print epoch
	optimizer.zero_grad()
	for dat, tar in train_loader: 
		structure = Variable(dat.view(-1, 3))
		permeability = Variable(tar.view(-1,1))
		#forward+backprop
		
		outputs = net.feed_forward(structure)
		loss = criterion(outputs, permeability)
		loss.backward()
	optimizer.step()

	for val_d, val_t in val_loader:
		v_structure = Variable(val_d.view(-1, 3))
		v_label = Variable(val_t.view(-1,1))

		out_v = net.feed_forward(v_structure)
		v_loss = criterion(out_v, v_label)


	#print('Epoch %s'% epoch, 'Train_Loss %s'% loss.data[0],
		#'Val_loss %s' % v_loss.data[0])
	train_err.append(loss.data[0])
	val_err.append(v_loss.data[0])
#print train_err, val_err
# test the model
net.eval()
test_err = []
for test_dat, test_tar in test_loader:
	t_structure = Variable(test_dat.view(-1, 3))
	label = Variable(test_tar.view(-1,1))

	out = net.feed_forward(t_structure)
	diff = label - out
	mae = np.mean(np.absolute(diff.data.numpy()))

	test_err.append(mae)

	print('Test Error %s' % np.mean(np.array(test_err)))



# evaluate on testset 



plt.plot(range(num_epoch), train_err, color = 'blue', linewidth = 3, label = 'Training Error')
plt.plot(range(num_epoch), val_err,  color = 'red', linewidth = 1.5, label = 'Validation Error')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
plt.ylabel('MSE')
plt.show()






#plt.plot(data, lables_sub,'ro')

# linear regression
'''regr = linear_model.LinearRegression()
regr.fit(np.array(para).reshape(-1,1), lables_sub)
y = regr.predict(np.array(para).reshape(-1,1))
print regr.coef_
plt.plot(para, lables_sub,'ro')
plt.plot(para, y, color = 'blue', linewidth = 3)
plt.xlabel('e^3/1+e')
plt.ylabel('permeability')
plt.show()

print("Mean squared error: %.2f"
      % mean_squared_error(lables_sub, y))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(lables_sub, y))
'''




print("---%s seconds---" % (time.time() - start))
