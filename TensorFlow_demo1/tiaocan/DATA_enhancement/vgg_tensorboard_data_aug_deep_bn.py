import tensorflow as tf
import os
import pickle
import numpy as  np

CIFAR_DIR = "D:/user/Download/cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))

#thensorBoard
#1,指定面板图上显示的变量
#2.在训练过程中将这些变量值计算出来写到文件中
#3，文件解析 ./tensorboard -- logdir =dir

def load_data(filename):
    """ read data from data file."""
    with open(filename,'rb') as f:
        data = pickle.load(f ,encoding='latin1') #,encoding='latin1',encoding='iso-8859-1'，encoding='bytes'
        return data['data'],data['labels']
# tensorflow.dataset(可以使用)
class cifardata:
    def __init__(self,filenames,need_shuffle):
        all_data = []
        all_labels=[]
        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)

        self._data =np.vstack(all_data)
        self._data = self._data
        self._labels = np.hstack(all_labels)
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()
    def _shuffle_data(self):
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]
    def next_batch(self,batch_size):
        '''return batch_size examples as a batch.'''
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator:end_indicator]
        batch_lables = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data,batch_lables

train_filenames = [os.path.join(CIFAR_DIR,'data_batch_%d' % i ) for i in range(1,6)]
test_filenames  = [os.path.join(CIFAR_DIR,'test_batch')]

train_data = cifardata(train_filenames,True)
test_data = cifardata(test_filenames,False)


batch_size =20
x = tf.placeholder(tf.float32, [batch_size,3072])
y = tf.placeholder(tf.int64,[batch_size])
is_training = tf.placeholder(tf.bool,[])
x_image = tf.reshape(x,[-1,3,32,32])
x_image = tf.transpose(x_image,perm=[0,2,3,1])
x_image_arr = tf.split(x_image,num_or_size_splits=batch_size,axis=0)
result_x_image_arr = []
for x_single_image in x_image_arr:
    x_single_image = tf.reshape(x_single_image,[32,32,3])
    data_aug_1=tf.image.random_flip_left_right(x_single_image)
    data_aug_2 = tf.image.random_brightness(data_aug_1,max_delta=63)
    data_aug_3 = tf.image.random_contrast(data_aug_2,lower=0.2,upper=1.8)
    x_single_image = tf.reshape(data_aug_3,[1,32,32,3])
    result_x_image_arr.append(x_single_image)
result_x_images = tf.concat(result_x_image_arr,axis=0)
normal_result_x_images = result_x_images /127.5 -1
#feature_map ,输出图 ，神经元图
'''
def conv_wrapper(inputs,
                 name,
                 output_channel = 32,
                 kernel_size = (3,3),
                 activation = tf.nn.relu,
                 padding = 'same'):
    """wrapper of tf.layers.conv2d"""
    return tf.layers.conv2d(inputs,
                            output_channel,
                            kernel_size,
                            padding = padding,
                            activation = activation,
                            name = name)
'''

def conv_wrapper(inputs,
                 name,
                 is_training,
                 output_channel = 32,
                 kernel_size = (3,3),
                 activation = tf.nn.relu,
                 padding = 'same'):
    """wrapper of tf.layers.conv2d"""
    #with batch normalization :conv -> bn -> activation
    with tf.name_scope(name):
      conv2d = tf.layers.conv2d(inputs,
                                output_channel,
                                kernel_size,
                                padding = padding,
                                activation = None,
                                name =name + '/conv2d')
      bn = tf.layers.batch_normalization(conv2d,
                                         training = is_training)
      return activation(bn)

def pooling_wrapper(inputs,name):
    return tf.layers.max_pooling2d(inputs,
                                   (2,2),
                                   (2,2),
                                   name = name)
conv1_1 = conv_wrapper(normal_result_x_images,'conv1_1',is_training)
conv1_2 = conv_wrapper(conv1_1,'conv1_2',is_training)
conv1_3 = conv_wrapper(conv1_2,'conv1_3',is_training)
pooling1 = pooling_wrapper(conv1_3,'pool1')

conv2_1 = conv_wrapper(pooling1,'conv2_1',is_training)
conv2_2 = conv_wrapper(conv2_1,'conv2_2',is_training)
conv2_3 = conv_wrapper(conv2_2,'conv2_3',is_training)
pooling2 = pooling_wrapper(conv2_3,'pool2')

conv3_1 = conv_wrapper(pooling2,'conv3_1',is_training)
conv3_2 = conv_wrapper(conv3_1,'conv3_2',is_training)
conv3_3 = conv_wrapper(conv3_2,'conv3_3',is_training)
pooling3 = pooling_wrapper(conv3_3,'pool3')
#16*16

#[none , 4*4*32]
flatten = tf.layers.flatten(pooling3)
y_ = tf.layers.dense(flatten,10)


loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)
#y_ ->sofmax
#y ->one_hot
#loss->ylogy_
predict = tf.argmax(y_,1)
correct_prediction = tf.equal(predict,y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# def variable_summary(var,name):
#      with tf.name_scope(name):
#          mean = tf.reduce_mean(var)
#          with tf.name_scorp('stddev'):
#              stddev =tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#          tf.summary.scalar('mean',mean)
#          tf.summary.scalar('stddev',stddev)
#          tf.summary.scalar('min',tf.reduce_min(var))
#          tf.summary.scalar('max',tf.reduce_max(var))
#          tf.summary.scalar('histogram',var)
#
# with tf.name_scope('summary'):
#     variable_summary(conv1_1,'conv1_1')
#     variable_summary(conv1_2,'conv1_2')
#     variable_summary(conv2_1,'conv2_1')
#     variable_summary(conv2_2,'conv2_2')
#     variable_summary(conv3_1,'conv3_1')
#     variable_summary(conv3_2,'conv3_2')


loss_summary = tf.summary.scalar('loss',loss)
accuracy_summary = tf.summary.scalar('accuracy',accuracy)
inputs_summary = tf.summary.image('input_image',normal_result_x_images)
merged_summary = tf.summary.merge_all()
merged_summary_test = tf.summary.merge([loss_summary,accuracy_summary])

LOG_DIR='.'
run_label = 'run_vgg_tensorboard'
run_dir = os.path.join(LOG_DIR,run_label)
if not os.path.exists(run_dir):
    os.mkdir(run_dir)
train_log_dir = os.path.join(run_dir,'train')
test_log_dir = os.path.join(run_dir,'test')
if not os.path.exists(train_log_dir):
    os.mkdir(train_log_dir)
if not os.path.exists(test_log_dir):
    os.mkdir(test_log_dir)


init = tf.global_variables_initializer()

train_steps = 1000
test_steps = 100
output_summary_every_steps = 100
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(train_log_dir,sess.graph)
    test_writer = tf.summary.FileWriter(test_log_dir)

    fixed_test_batch_data,fixed_test_batch_labels \
       = test_data.next_batch(batch_size)
    for i in range(train_steps):
        batch_data,batch_labels = train_data.next_batch(batch_size)
        eval_ops = [loss,accuracy,train_op]
        should_output_summary =((1+i) % output_summary_every_steps ==0)
        if should_output_summary:
            eval_ops.append(merged_summary)

        eval_ops_results = sess.run(
            eval_ops,
            feed_dict={
                x:batch_data ,
                y:batch_labels,
                is_training:True})
        loss_val,acc_val = eval_ops_results[0:2]
        if should_output_summary:
            train_summary_str = eval_ops_results[-1]
            train_writer.add_summary(train_summary_str,i+1)
            test_summary_str = sess.run([merged_summary_test],
                                        feed_dict={
                                            x:fixed_test_batch_data,
                                            y:fixed_test_batch_labels,
                                            is_training : False
                                            })[0]
            test_writer.add_summary(test_summary_str,i+1)

        if (i+1) % 100 == 0:
            print('[train] step: %d ,loss:%4.5f, acc: %4.5f' %(i,loss_val,acc_val))
        if (i+1) % 100 == 0:
            test_data = cifardata(test_filenames, False)
            all_test_acc_val = [ ]
            for j in range(test_steps):
                test_batch_data, test_batch_labels \
                    = test_data.next_batch(batch_size)
                test_acc_val = sess.run(
                    [accuracy],
                    feed_dict={
                        x:test_batch_data,
                        y:test_batch_labels,
                        is_training: False})
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[test]step: %d ,acc:%4.5f '%(i+1,test_acc ) )