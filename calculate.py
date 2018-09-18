pos_lines = open('/lfs1/users/szhu/project/MTCNN-keras/data4/12/label-train.txt','r')
lines = len(pos_lines.readlines())
print('pos num:',lines)
neg_lines = open('/lfs1/users/szhu/project/MTCNN-keras/data3/12/neg_12.txt','r')
lines = len(neg_lines.readlines())
print('neg num:',lines)
part_lines = open('/lfs1/users/szhu/project/MTCNN-keras/data3/12/part_12.txt','r')
lines = len(part_lines.readlines())
print('part num:',lines)


