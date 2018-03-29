import os
import tensorflow as tf
import ConstantDefiner

def generate_image_and_label():
    filenames = [os.path.join(ConstantDefiner.DATA_DIR, 'mnist_batch%d.csv' % i)
                 for i in range(6)]
    filename_queue = tf.train.string_input_producer(filenames)
    image_bytes = ConstantDefiner.IMAGE_HEIGHT * ConstantDefiner.IMAGE_WIDTH
    record_bytes = image_bytes + ConstantDefiner.LABEL_BYTES
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_default = []
    for i in range(record_bytes):
        record_default.append([1])
    records= tf.decode_csv(value,record_defaults=record_default)
    image = tf.reshape(tf.cast(tf.strided_slice(records, [0], [image_bytes]), tf.float32),
                       [ConstantDefiner.IMAGE_HEIGHT, ConstantDefiner.IMAGE_WIDTH, 1])
    image.set_shape([ConstantDefiner.IMAGE_HEIGHT, ConstantDefiner.IMAGE_WIDTH, 1])
    label = tf.cast(tf.strided_slice(records, [image_bytes], [record_bytes]), tf.int32)
    label.set_shape([1])
    return image,label

# 产生图片队列
def generate_images_and_labels_batch(image, label, shuffle):
    label = tf.reshape(tf.one_hot(label,depth=ConstantDefiner.NUM_CLASSES,axis=0),[10])
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(min_fraction_of_examples_in_queue * ConstantDefiner.NUM_EXAMPLES_FOR_TRAIN)
    num_preprocess_threads = 16
    if shuffle:
        images, labels = tf.train.shuffle_batch(
            [image, label], batch_size=ConstantDefiner.BATCH_SIZE,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * ConstantDefiner.BATCH_SIZE,
            min_after_dequeue=min_queue_examples)
    else:
        images, labels = tf.train.shuffle_batch(
            [image, label], batch_size=ConstantDefiner.BATCH_SIZE,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples
        )
    return images, labels


