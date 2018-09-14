#coding:utf-8
import numpy as np

def IoU2(box, boxes):
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
    maximum = max(ovrs)
    location = ovrs.index(maximum)
    return maximum,location
