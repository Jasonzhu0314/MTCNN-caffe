import sys
from numpy import random

# devide the total data into train and val 
# select the (-neg_val_num/3 - 0) as the val data  


def data_separate(img_dir,path,size):
      global neg_val_num
      all_txt= img_dir + size +'/' + path + size + '.txt'
      train_txt= img_dir + size +'/' + path + size + '_train.txt'
      val_txt= img_dir + size +'/' + path + size + '_val.txt'
      val = open(val_txt, 'w')
      train = open(train_txt, 'w')
      with open(all_txt, 'r') as a:
            lines = a.readlines()
            random.shuffle(lines)
            for i in range(int(len(lines) * 0.999)):
                  p = lines[i].find(" ") + 1
                  if path == "pos_" or path == "part_":
                        string = img_dir + lines[i][:p-1] + ".jpg " + lines[i][p:-1] + "\n"
                  else:
                        string = img_dir + lines[i][:p-1] + ".jpg " + lines[i][p:-1] + " -1 -1 -1 -1\n"
                  train.write(string)
                  if i % 2000000 == 0:
                      print(i)
            if path == "neg_":
                  neg_val_num = int(len(lines)*0.001)
                  start = -neg_val_num
            else:
                  start = -int(neg_val_num/3)
            for i in range(start,0):
                  p = lines[i].find(" ") + 1
                  if path == "pos_" or path == "part_":
                        string = img_dir + lines[i][:p-1] + ".jpg " + lines[i][p:-1] + "\n"
                  else:
                        string = img_dir + lines[i][:p-1] + ".jpg " + lines[i][p:-1] + " -1 -1 -1 -1\n"
                  val.write(string)
      val.close()
      train.close()

size = sys.argv[1]
img_dir = "/home/users/zhuzhengshuai/data/mtcnn/data/"



txt_lists = ["neg_","pos_","part_"]
for path in txt_lists:
      data_separate(img_dir,path,size)