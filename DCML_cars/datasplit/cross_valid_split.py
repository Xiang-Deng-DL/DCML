import re
with open('./cars_train.txt', 'r') as src:
	srclines = src.readlines()

with open('./cars_train1.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		if int(label) < 74:#total 98 classes, [0, 74] as training
			tf.write(line)
      
with open('./cars_valid1.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())		
		if int(label) >= 74:#[74, 98) as valid
			tf.write(line)            

with open('./cars_train2.txt', 'w') as tf:
	for line in srclines:
		ipath, label = re.split(r",| ", line.strip())
		label = int(label)
		if label < 50 or label>=74:#50+24=74
			tf.write(line)
      
with open('./cars_valid2.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		label = int(label)
		if label >= 50 and label<74:#[50, 74) as valid
			tf.write(line) 
         
with open('./cars_train3.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		label = int(label)
		if label < 26 or label>=50:#26+48=74
			tf.write(line)
      
with open('./cars_valid3.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		label = int(label)
		if label >= 26 and label<50:#[26, 50) as valid
			tf.write(line)
          
with open('./cars_train4.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		label = int(label)
		if label >= 24:
			tf.write(line)
      
with open('./cars_valid4.txt', 'w') as tf:
	for line in srclines:
         path, label = re.split(r",| ", line.strip())
         label = int(label)
         if label<24:#[0, 24) as valid
             tf.write(line)
