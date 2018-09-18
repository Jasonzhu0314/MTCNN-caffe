import os
def generate_annotation(wider_path,annotation_path,image_path):
    #small_data_path = '/lfs1/users/szhu/project/MTCNN-keras/label/small_data.txt'
    #small_data = open(small_data_path,'w') 
    i = 0
    with open(annotation_path,'w') as annotations:
        with open(wider_path,'r') as widers:
            wider = widers.readlines()
            while i < len(wider):
                path = os.path.join(image_path+wider[i].strip())
                #small_data.write(image_path)
                i +=1
                num = int(wider[i])
                annotations.write(path)
                annotations.write(' ')
                for j in range(num):
                    i +=1
                    coordinates = wider[i].strip().split(' ')[:4]
                    annotations.write(coordinates[0])
                    annotations.write(' ')
                    annotations.write(coordinates[1])
                    annotations.write(' ')
                    annotations.write(str(int(coordinates[0])+int(coordinates[2])))
                    annotations.write(' ')
                    annotations.write(str(int(coordinates[1])+int(coordinates[3])))
                    annotations.write(' ')                      
                annotations.write('\n')
                i +=1
    #small_data.close()
if __name__ == '__main__':
    image_path = '/lfs1/users/szhu/data/Face_detection/WIDER_train/images/'
    wider_path = '/lfs1/users/szhu/project/MTCNN-keras/label/wider_face_train_bbx_gt.txt'
    small_data_path = '/lfs1/users/szhu/project/MTCNN-keras/label/wider_gt.txt'
    generate_annotation(wider_path,small_data_path,image_path)
