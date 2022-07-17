import re



#class_num 11318
#img_num 59551
#Training: 8489 valid: 2829

with open('./sop_train.txt', 'r') as src:
	srclines = src.readlines()

with open('./sop_train1.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		if int(label) < 8489:##0-8489 
			tf.write(line)

with open('./sop_valid1.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())		
		if int(label) >= 8489:#[8489, 11318) as valid
			tf.write(line)


with open('./sop_train2.txt', 'w') as tf:
	for line in srclines:
		ipath, label = re.split(r",| ", line.strip())
		label = int(label)
		if label < 5660 or label>=8489:
			tf.write(line)
      
with open('./sop_valid2.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		label = int(label)
		if label >= 5660 and label<8489:#
			tf.write(line)    


with open('./sop_train3.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		label = int(label)
		if label < 2831 or label>=5660:
			tf.write(line)
      
with open('./sop_valid3.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		label = int(label)
		if label >= 2831 and label<5660:#
			tf.write(line)


with open('./sop_train4.txt', 'w') as tf:
	for line in srclines:
		path, label = re.split(r",| ", line.strip())
		label = int(label)
		if label >= 2829:
			tf.write(line)

with open('./sop_valid4.txt', 'w') as tf:
	for line in srclines:
         path, label = re.split(r",| ", line.strip())
         label = int(label)
         if label<2829:#
             tf.write(line)
