import tensorflow as tf
import os
import pickle
import numpy as  np

CIFAR_DIR = "D:/user/Download/cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))

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
            # for item, label in zip(data,labels):
            #     if label in [0,1]:
            #         all_data.append(item)
            #         all_labels.append(label)
        self._data =np.vstack(all_data)
        self._data = self._data/127.5 - 1 #数据归一化
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



x = tf.placeholder(tf.float32, [None,3072])
y = tf.placeholder(tf.int64,[None])
w = tf.get_variable('w',[x.get_shape()[-1],10],
                    initializer=tf.random_normal_initializer(0,1))
b = tf.get_variable('b',[10],
                    initializer=tf.constant_initializer(0.0))
y_ = tf.matmul(x , w) + b


#mean square loss
"""
p_y = tf.nn.softmax(y_) #softmax API : e^x/sum(e^x)
# one_hot 编码
y_one_hot = tf.one_hot(y,10,dtype=tf.float32)
loss = tf.reduce_mean(tf.square(y_one_hot - p_y))
"""
loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)
#y_ ->sofmax
#y ->one_hot
#loss->ylogy_


"""p_y_1 = tf.nn.sigmoid(y_)
y_reshaped = tf.reshape(y,(-1,1))
y_reshaped_float = tf.cast(y_reshaped,tf.float32)
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))
""" #多分类不使用sigmoid
predict = tf.argmax(y_,1)
correct_prediction = tf.equal(predict,y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100
with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data,batch_labels = train_data.next_batch(batch_size)
        loss_val,acc_val,_ = sess.run(
            [loss,accuracy,train_op],
            feed_dict={
                x:batch_data ,
                y:batch_labels })
        if (i+1) % 5000 == 0:
            print('[train] step: %d ,loss:%4.5f, acc: %4.5f' %(i,loss_val,acc_val))
        if (i+1) % 5000 == 0:
            test_data = cifardata(test_filenames, False)
            all_test_acc_val = [ ]
            for j in range(test_steps):
                test_batch_data, test_batch_labels \
                    = test_data.next_batch(batch_size)
                test_acc_val = sess.run(
                    [accuracy],
                    feed_dict={
                        x:test_batch_data,
                        y:test_batch_labels})
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[test]step: %d ,acc:%4.5f '%(i+1,test_acc ) )

