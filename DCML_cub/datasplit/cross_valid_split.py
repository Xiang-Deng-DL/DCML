import re
#cub_train.txt is generated from split_train_test.py
with open('./cub_train.txt', 'r') as src:
	srclines = src.readlines()

#cross-validation: split the training data samples into training and validation datasets; validation datasets (25%) are used for hyper-parameter tuning
with open('./cub_train1.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		if int(label) < 75:#total 100 classes; [0, 75) as train
			tf.write(line)
			#print(line, file=tf)
      
with open('./cub_valid1.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())		
		if int(label) >= 75:#[75, 100) as valid
			tf.write(line)
			#print(line, file=tf)
            


with open('./cub_train2.txt', 'w') as tf:
	for line in srclines:
		ipath, label = re.split(r",| ", line.strip())
		label = int(label)
		if label < 50 or label>=75:
			tf.write(line)
			#print(line, file=tf)
      
with open('./cub_valid2.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		label = int(label)
		if label >= 50 and label<75:#[50, 75) as valid
			tf.write(line)
			#print(line, file=tf)
            

            
with open('./cub_train3.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		label = int(label)
		if label < 25 or label>=50:
			tf.write(line)
			#print(line, file=tf)
     
with open('./cub_valid3.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		label = int(label)
		if label >= 25 and label<50:#[25, 50) as valid
			tf.write(line)
			#print(line, file=tf)
            
            

with open('./cub_train4.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		label = int(label)
		if label >= 25:
			tf.write(line)
			#print(line, file=tf)
     
with open('./cub_valid4.txt', 'w') as tf:
	for line in srclines:
         path, label = re.split(r",| ", line.strip())
         label = int(label)
         if label<25:#[0, 25) as valid
             tf.write(line)
             #print(line, file=tf)
            