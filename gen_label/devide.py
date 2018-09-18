from numpy import random
def data_separate():
      all_txt='/lfs1/users/szhu/project/MTCNN-keras/data/12/label.txt'
      train_txt='/lfs1/users/szhu/project/MTCNN-keras/data/12/label-train.txt'
      test_txt='/lfs1/users/szhu/project/MTCNN-keras/data/12/label-test.txt'
      test = open(test_txt,'w')
      train = open(train_txt,'w')
      with open(all_txt,'r') as a:
            lines = a.readlines()
            random.shuffle(lines)
            for i in range(int(len(lines)*0.9)):
                  train.write(lines[i])
                  if i % 1000 == 0:
                      print i
            for i in range(int(len(lines)*0.9), int(len(lines))):
                  test.write(lines[i])
      test.close()
      train.close()
data_separate()