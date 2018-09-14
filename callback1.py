#coding: utf-8
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
'''
if K.backend() == 'tensorflow':
    import tensorflow as tf
    from tensorflow.contrib.tensorboard.plugins import projector
'''
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
        bacth_loss_path = '/lfs1/users/szhu/project/MTCNN-keras/loss/batch_0.01_1_loss.txt'
        with open(bacth_loss_path,'a') as batch_loss:
            batch_loss.write(str(logs.get('cla_loss')))
            batch_loss.write(' ')
            batch_loss.write(str(logs.get('cla_acc')))
            batch_loss.write(' ')
            batch_loss.write(str(logs.get('bbox_loss')))
            batch_loss.write(' ')
            batch_loss.write(str(logs.get('bbox_acc')))
            batch_loss.write(' ')
            batch_loss.write('\n')

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        epoch_loss_path = '/lfs1/users/szhu/project/MTCNN-keras/loss/epoch_0.01_1_loss.txt'
        with open(epoch_loss_path,'a') as epoch_loss:       
            epoch_loss.write(str(logs.get('cla_loss')))
            epoch_loss.write(' ')
            epoch_loss.write(str(logs.get('cla_acc')))
            epoch_loss.write(' ')
            epoch_loss.write(str(logs.get('bbox_loss')))
            epoch_loss.write(' ')
            epoch_loss.write(str(logs.get('bbox_acc')))
            epoch_loss.write('\n')
        validation_loss = '/lfs1/users/szhu/project/MTCNN-keras/loss/validation_0.01_1_acc.txt'
        with open(validation_loss,'a') as batch_loss:
            batch_loss.write(str(logs.get('val_cla_loss')))
            batch_loss.write(' ')
            batch_loss.write(str(logs.get('val_cla_acc')))
            batch_loss.write(' ')
            batch_loss.write(str(logs.get('val_bbox_loss')))
            batch_loss.write(' ')
            batch_loss.write(str(logs.get('val_bbox_acc')))
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

class TBloss(keras.callbacks.Callback):
    """Tensorboard basic visualizations.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    TensorBoard is a visualization tool provided with TensorFlow.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by Tensorboard.
        histogram_freq: frequency (in epochs) at which to compute activation
            histograms for the layers of the model. If set to 0,
            histograms won't be computed.
        write_graph: whether to visualize the graph in Tensorboard.
            The log file can become quite large when
            write_graph is set to True.
        write_images: whether to write model weights to visualize as
            image in Tensorboard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 write_batch_performance=True,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TBloss, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_grads = write_grads
        self.write_graph = write_graph
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size
        self.seen = 0
        self.write_batch_performance = write_batch_performance

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = w_img.get_shape()
                        if len(shape) > 1 and shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)
                        if len(shape) == 1:
                            w_img = tf.expand_dims(w_img, 0)
                        w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)
                        tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq:
            self.saver = tf.train.Saver()

            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']

            embeddings = {layer.name: layer.weights[0]
                          for layer in self.model.layers
                          if layer.name in embeddings_layer_names}

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings.keys()}

            config = projector.ProjectorConfig()
            self.embeddings_logs = []

            for layer_name, tensor in embeddings.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                self.embeddings_logs.append(os.path.join(self.log_dir,
                                                         layer_name + '.ckpt'))

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    batch_val = []
                    batch_val.append(val_data[0][i:i + step])
                    batch_val.append(val_data[1][i:i + step])
                    batch_val.append(val_data[2][i:i + step])
                    if self.model.uses_learning_phase:
                        batch_val.append(val_data[3])
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, self.seen)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.seen)
        self.writer.flush()
        self.seen += self.batch_size

    def on_train_end(self, _):
        self.writer.close()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if self.write_batch_performance == True:
            for name, value in logs.items():
                if name in ['batch','size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.seen)
            self.writer.flush()

        self.seen += self.batch_size

#创建一个实例LossHistory
#history = LossHistory()