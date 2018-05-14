# MTCNN-keras
## code description
### gen_12_data.py
生成p-Net网络需要的positive，negative，part人脸
### PNet.py
PNet网络结构
### gen_list.py
生成训练的标签，将positive，negative，part图片的路径和标签打乱放在一起
### utils.py
IOU,label_convert 函数
### callback.py
训练过程过程中的回调函数
