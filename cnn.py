import tensorflow as tf
import ConstantDefiner

#CNN算法卷积层，池化层，
def inference(images):
    #conv1
    kernel = weight_variable('weights1',[5,5,1,32])
    conv = tf.nn.conv2d(images,kernel,[1,1,1,1],padding='SAME')
    bias = bias_variable('bias1',[32])
    conv1 = tf.nn.relu(tf.nn.bias_add(conv,bias))
    #pool1
    pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1')


    #conv2
    kernel = weight_variable('weights2',[5,5,32,64])
    conv = tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
    bias = bias_variable('bias2',[64])
    conv2 = tf.nn.relu(tf.nn.bias_add(conv,bias))
    pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2')

    #full3
    reshape = tf.reshape(pool2,[ConstantDefiner.BATCH_SIZE,-1])
    # dim = reshape.get_shape()[1].value
    weights = weight_variable('weights3',[7*7*64,1024],stddev=0.04)
    bias = bias_variable('bias3',[1024],value=0.1)
    full3 = tf.nn.relu(tf.matmul(reshape,weights)+bias)

    #full4
    h_fc1_drop = tf.nn.dropout(full3, 0.5)
    weights = weight_variable('weights5',[1024,ConstantDefiner.NUM_CLASSES],stddev=1/192.0)
    bias = bias_variable('bias5',[ConstantDefiner.NUM_CLASSES])
    linear_result = tf.add(tf.matmul(h_fc1_drop,weights),bias)
    return linear_result

#计算损失
def loss(linear_result,labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels,logits=linear_result)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean

#权重
def weight_variable(name,shape,stddev=5e-2):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(name=name,shape=shape,initializer=initializer)
#偏向
def bias_variable(name,shape,value=0.0):
    initializer = tf.constant_initializer(value=value)
    return tf.get_variable(name=name,shape=shape,initializer=initializer)
