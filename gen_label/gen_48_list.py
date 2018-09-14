import sys
import os
import numpy as np
from numpy import random
save_dir = "../data4/48"
img_dir = "/lfs1/users/szhu/project/MTCNN-keras/data4/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
f1 = open(os.path.join(save_dir, 'pos_48.txt'), 'r')
f2 = open(os.path.join(save_dir, 'neg_48.txt'), 'r')
f3 = open(os.path.join(save_dir, 'part_48.txt'), 'r')

pos = f1.readlines()
neg = f2.readlines()
part = f3.readlines()
np.random.shuffle(pos)
np.random.shuffle(neg)
np.random.shuffle(part)
pos_lines = []
neg_lines = []
part_lines = []
lines = []

num = 7132  # data4
#num = 2000

for i in range(192308):
    p = pos[i].find(" ") + 1
    #pos[i] = img_dir + pos[i][:p-1] + ".jpg " + pos[i][p:-1] + "\n"
    pos[i] = img_dir + pos[i][:p-1] + ".jpg " + pos[i][p:-1] + "\n"
    lines.append(pos[i])
    #f.write(pos[i])

for i in range(137032):
    p = neg[i].find(" ") + 1
    #neg[i] = img_dir + neg[i][:p-1] + ".jpg " + neg[i][p:-1] + " -1 -1 -1 -1\n"
    neg[i] = img_dir + neg[i][:p-1]+ ".jpg " + neg[i][p:-1] + " -1 -1 -1 -1\n"
    lines.append(neg[i])
    #f.write(neg[i])

for i in range(126973):
    p = part[i].find(" ") + 1
    #part[i] = img_dir + part[i][:p-1] + ".jpg " + part[i][p:-1] + "\n" # add file attribute to the string
    part[i] = img_dir + part[i][:p-1] + ".jpg " + part[i][p:-1] + "\n"
    lines.append(part[i])
    #f.write(part[i])
'''
np.random.shuffle(pos_lines)
np.random.shuffle(neg_lines)
for i in range(num):
    lines.append(neg_lines[i])
    lines.append(pos_lines[i])
    lines.append(neg_lines[num*1+i])
    lines.append(part_lines[i])
'''
np.random.shuffle(lines)
train_txt='/lfs1/users/szhu/project/MTCNN-keras/data4/48/label-train.txt'
test_txt='/lfs1/users/szhu/project/MTCNN-keras/data4/48/label-test.txt'
test = open(test_txt,'w')
train = open(train_txt,'w')
for i in range(int(len(lines)*0.9)):
    train.write(lines[i])
    #if i % 1000 == 0:
    #    print i
print i
for i in range(int(len(lines)*0.9), int(len(lines))):
    test.write(lines[i])
print i
test.close()
train.close()
f1.close()
f2.close()
f3.close()

