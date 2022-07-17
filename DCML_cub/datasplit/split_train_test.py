from scipy.io import loadmat


# CUB200-2011
with open('./CUB_200_2011/images.txt', 'r') as src:
	srclines = src.readlines()

with open('./CUB_200_2011/cub_train.txt', 'w') as tf:
	for line in srclines:
		i, fname = line.strip().split()
		label = int(fname.split('.', 1)[0])
		if label <= 100:
			print('images/{},{}'.format(fname, label-1), file=tf) 
      
with open('./CUB_200_2011/cub_test.txt', 'w') as tf:
	for line in srclines:
		i, fname = line.strip().split()
		label = int(fname.split('.', 1)[0])
		if label > 100:
			print('images/{},{}'.format(fname, label-1), file=tf)
