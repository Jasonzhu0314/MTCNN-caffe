#coding:utf-8
import numpy as np

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    ovrs = []
    for i in range(boxes.shape[0]):
        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        area = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1)
        xx1 = np.maximum(box[0], boxes[i, 0])
        yy1 = np.maximum(box[1], boxes[i, 1])
        xx2 = np.minimum(box[2], boxes[i, 2])
        yy2 = np.minimum(box[3], boxes[i, 3])
    # 计算这两个框重叠起来的坐标
    # compute the width and height of the bounding box 重叠起来的长和宽
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h # 重叠部分的大小
        
        ovr = inter / (box_area + area - inter) # 重叠部分/两个框部分相加-重叠部分
        ovrs.append(ovr)
    #print ovrs
    minimum = min(ovrs)
    return minimum


def IoU1(box,boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    ovrs = []
    for i in range(boxes.shape[0]):
        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        area = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1)
        xx1 = np.maximum(box[0], boxes[i, 0])
        yy1 = np.maximum(box[1], boxes[i, 1])
        xx2 = np.minimum(box[2], boxes[i, 2])
        yy2 = np.minimum(box[3], boxes[i, 3])
    # 计算这两个框重叠起来的坐标
    # compute the width and height of the bounding box 重叠起来的长和宽
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h # 重叠部分的大小
        ovr = inter / (box_area + area - inter) # 重叠部分/两个框部分相加-重叠部分
        ovrs.append(ovr)
    #print(ovrs)
    maximum = max(ovrs)
    return maximum

def IoU2(box,boxes):
    index = []
    ovrs = []
    for i in range(boxes.shape[0]):
        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        area = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1)
        xx1 = np.maximum(box[0], boxes[i, 0])
        yy1 = np.maximum(box[1], boxes[i, 1])
        xx2 = np.minimum(box[2], boxes[i, 2])
        yy2 = np.minimum(box[3], boxes[i, 3])
    # 计算这两个框重叠起来的坐标
    # compute the width and height of the bounding box 重叠起来的长和宽
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h # 重叠部分的大小
        ovr = inter / (box_area + area - inter) # 重叠部分/两个框部分相加-重叠部分
        ovrs.append(ovr)

    #print(ovrs)
    maximum = max(ovrs)
    location = ovrs.index(maximum)
    return maximum,location

def convert_to_square(bbox):
    """Convert bbox to square

    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox

    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h,w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox

def convert_cla(y,num_classes):
    # 因为全卷积网络最后的输出是4维数组，所以要扩展相应的维度，以适应输出
    
    n = y.shape[0]
    categorical = np.zeros((n,1,1,num_classes))
    for i in range(n):
        categorical[i,0,0,:] = y[i]
    #print(categorical)
    #print(categorical.shape)
    #categorical[np.arange(n),0,0, y] = 1
    return categorical

def convert_bbox(y,num):
    n = y.shape[0]
    bbox = np.zeros((n,1,1,num))
    for i in range(n):
        bbox[i,0,0,:] = y[i]
    return bbox


if __name__ == '__main__':
    #y = np.array([1,0,0,1])
    #y1=convert_cla(y,2)
    y =np.array([[1,1,0],[1,0,1],[0,0,-1]])
    y1 = convert_cla(y,3) 
    y = np.array([[1,0.2,0.4,0.5,0.6],[0,-1,-1,-1,-1]])
    y2 = convert_bbox(y,5)
    print(y2)
    print(y1)
    if y1[0,0,0,0] == -1:
        print('x')

    #j = [0,1,2]
    #j.append(3)
    #i = np.array(j)
    #x = y1[i,:,:,:]
    #print(x)
    '''
    print(y1.shape[0])
    x1 = y1[2,:,:,:]
    x = np.zeros((1,1,2))
    print(x1)
    print(x)
    '''