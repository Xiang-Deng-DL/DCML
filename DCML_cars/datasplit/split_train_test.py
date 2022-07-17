from scipy.io import loadmat

# Cars196
file = loadmat('./CARS196/cars_annos.mat')
annos = file['annotations']

with open('./CARS196/cars_train.txt', 'w') as tf:
    for i in range(16185):
        if annos[0,i][-2] <= 98:
            print('{},{}'.format(annos[0,i][0][0], annos[0,i][-2][0][0]-1), file=tf)

with open('./CARS196/cars_test.txt', 'w') as tf:
    for i in range(16185):
        if annos[0,i][-2] > 98:
            print('{},{}'.format(annos[0,i][0][0], annos[0,i][-2][0][0]-1), file=tf)
