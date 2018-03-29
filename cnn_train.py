
import tensorflow as tf
import cnn_input as cnn_input
import cnn as cnn
import time

image,label = cnn_input.generate_image_and_label()
images, labels = cnn_input.generate_images_and_labels_batch(image=image, label=label, shuffle=True)
#神经网络计算出来的值
logits = cnn.inference(images)
loss = cnn.loss(logits, labels)  # 返回的交叉熵的均值
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)    #梯度下降
correct_predict = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))#在训练集上的正确率
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

t1 = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10000):
        if i %100==0:
            acc = sess.run(accuracy)
            print('epoch:%d, acc: %f' % (i, acc))
        train_op = sess.run(train_step)
    coord.request_stop()
    coord.join(threads)

