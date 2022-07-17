from scipy.io import loadmat


# SOP
with open('./SOP/Stanford_Online_Products/Ebay_train.txt', 'r') as src:
    srclines = src.readlines()

with open('./SOP/sop_train.txt', 'w') as tf:
    for i in range(1, len(srclines)):
        line = srclines[i]
        line_split = line.strip().split(' ')
        cls_id = str(int(line_split[1]) - 1)
        img_path = 'Stanford_Online_Products/'+line_split[3]
        print(img_path+','+cls_id, file=tf)

with open('./SOP/Stanford_Online_Products/Ebay_test.txt', 'r') as src:
    srclines = src.readlines()

with open('./SOP/sop_test.txt', 'w') as tf:
    for i in range(1, len(srclines)):
        line = srclines[i]
        line_split = line.strip().split(' ')
        cls_id = str(int(line_split[1]) - 1)
        img_path = 'Stanford_Online_Products/'+line_split[3]
        print(img_path+','+cls_id, file=tf)





