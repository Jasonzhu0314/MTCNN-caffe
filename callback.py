#coding: utf-8
import keras
import matplotlib.pyplot as plt
# create a custom callback by extending the base class keras.callbacks.Callback
class LossHistory(keras.callbacks.Callback):
    '''
    depiction：a simple class saving a list of losses over each batch during training
    params losse:training loss
           accuracy:training set accuracy      
           val_loss:valiation set loss
           val_acc:valiation set accuracy
           The type of all of them are directory including{'batch':[],'epoch':[]} 
    '''
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        '''
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        '''
        bacth_loss_path = '/lfs1/users/szhu/project/age_predict_DEX/batch_4_loss.txt'
        with open(bacth_loss_path,'a') as batch_loss:
            batch_loss.write(str(logs.get('loss')))
            batch_loss.write(' ')
            batch_loss.write(str(logs.get('acc')))
            batch_loss.write('\n')
        

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        epoch_loss_path = '/lfs1/users/szhu/project/age_predict_DEX/epoch_4_loss.txt'
        with open(epoch_loss_path,'a') as epoch_loss:       
            epoch_loss.write(str(logs.get('loss')))
            epoch_loss.write('\n')
        validation_loss = '/lfs1/users/szhu/project/age_predict_DEX/validation_4_acc.txt'
        with open(validation_loss,'a') as batch_loss:
            batch_loss.write(str(logs.get('val_loss')))
            batch_loss.write(' ')
            batch_loss.write(str(logs.get('val_acc')))
            batch_loss.write('\n')

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置
        plt.show()
    #def loss2txt(self,epoch_loss_path):
        
#创建一个实例LossHistory
#history = LossHistory()