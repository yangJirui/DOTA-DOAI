# -*- coding:utf-8 -*-
import os
'''
bridge 2075
baseball-diamond 412
container-crane 142
basketball-court 529
harbor 6016
soccer-ball-field 338
ground-track-field 331
small-vehicle 126686
plane 8072
storage-tank 5346
turntable 437
large-vehicle 22400
tennis-court 2438
helicopter 635
ship 32973
swimming-pool 2181
'''
LABEL_LIST = {}
def parse_label(txt_list):
    for i in txt_list[2:]:
        cat = i.strip().split()[8].strip()
        if cat not in LABEL_LIST:
            LABEL_LIST[cat] = 1
        else:
            LABEL_LIST[cat] +=1

raw_data = '/home/omnisky/DataSets/Dota/val'
raw_images_dir = os.path.join(raw_data, 'images', 'images')
raw_label_dir = os.path.join(raw_data, 'labelTxt_1.5', 'val')

save_dir = '/home/omnisky/DataSets/Dota_clip/trainval800/'

images = [i for i in os.listdir(raw_images_dir) if 'png' in i]
labels = [i for i in os.listdir(raw_label_dir) if 'txt' in i]

print ('find image', len(images))
print ('find label', len(labels))

min_length = 1e10
max_length = 1

for idx, img in enumerate(images):
    # img = 'P1524.png'
    print (idx, 'read image', img)
    txt_data = open(os.path.join(raw_label_dir, img.replace('png', 'txt')), 'r').readlines()

    # box = format_label(txt_data)
    parse_label(txt_data)
for key, item in LABEL_LIST.items():
    print key, item
