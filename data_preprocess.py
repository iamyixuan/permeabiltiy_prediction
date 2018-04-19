import numpy as np
import os
import time
import json

start = time.time()


filename = sorted(os.listdir('structure_data/data_ensem'))
filename.pop(0)

def prepocess_data(filename):

	l = []
	with open('structure_data/data_ensem/'+ str(filename)) as f:
		for line in f:
			l.append(line)
	new_list = []
	for i in l:
		new_list.append(take_out_white_spaces(i))
	return new_list


def take_out_white_spaces(x):
	a = x.split()
	lst = list(map(float, a))
	return lst


# save each sample as a single file in a directory

'''for i, file in enumerate(filename):
	out = prepocess_data(file)
	path = os.path.join('processed_data', filename[i] + '.npy')
	np.save(open(path,'w'), out)
	if i == 500:
		break
'''
indx = np.random.randint(0, 8300, size = 1000)
np.save(open('index.npy','w'), indx)




data_3d = []
for i in indx:
	mid_point = prepocess_data(filename[i])
	data_3d.append(mid_point)

data = np.array(data_3d)

with open('subset_rand.json','wb') as f:
	json.dump(data_3d, f)


print("---%s seconds---" % (time.time() - start))

