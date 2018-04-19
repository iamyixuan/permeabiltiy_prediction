import numpy as np
import pandas as pd
import json
import os
import itertools


'''
------------
This part is for generating void ratio for the whole sample
-------------
''' 
filename = sorted(os.listdir('data_ensem'))
filename.pop(0)
data_indx = np.load('index.npy','r')

def take_out_white_spaces(x):
	a = x.split()
	lst = list(map(float, a))
	return np.array(lst)

def prepocess_data(filename):

	l = []
	with open('data_ensem/'+ str(filename)) as f:
		for line in f:
			l.append(line)
	new_list = []
	for i in l:
		new_list.append(take_out_white_spaces(i))
	return new_list



'''void_ratio = []
for i in data_indx:
	f = prepocess_data(filename[i])
	num_0 = np.count_nonzero(f)
	void_r = (10000*100.0 - num_0)/ num_0
	void_ratio.append(void_r)

print len(void_ratio[:10])

with open('void_ratio.json','wb') as a:
	json.dump(void_ratio, a)'''




'''
------------------
This part is for extracting features for each line in each sample
and combine them togther as new sample features
-----------------
'''

def line_extract(matrix):
	features = []
	j = 0
	for i in range(1, 101):
		f = matrix[j:100*i]

		j = 100*i
		
		num_non = np.count_nonzero(f)
		void = (10000.0 - num_non)/num_non
		features.append(void)
	return features


extracted = []
for j in data_indx:
	f = prepocess_data(filename[j])

	extracted.append(line_extract(f))

print len(extracted)

np.save(open('extracted_feat.npy','w'), extracted)


