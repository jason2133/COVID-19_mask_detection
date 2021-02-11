import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import cv2

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

FLAGS = None
tf.logging.set_verbosity(tf.logging.ERROR)

img_w = 150
img_h = 200
img_c = 3
# 가로 width 150, 세로 height 200, color는 3으로 해서 BGR 먹인다.

# 3명 인식할꺼라 3이라고 해줌
num_cls = 3

# 훈련에 활용하는 폴더 위치
train_img_dir = 'data'

# 카테고리 3개
# correct - 마스크 제대로 씀
# incorrect - 마스크 제대로 안씀 (걸친 경우, 코스크, 턱스크)
# none - 마스크 안 씀

categories = ['correct', 'incorrect', 'none']
extensions = ['png', 'jpeg', 'jpg'] 

def create_image_list(image_dir):
    if not os.path.exists(image_dir):
        print('Error:', image_dir, 'is not exist!')
        return None
    
    image_list = []
    for label, category in enumerate(categories):
        filelist = os.listdir(os.path.join(image_dir, category))
        for f in filelist:
            dotext = os.path.splitext(f)[-1]
            # . 으로 구분

            ext = dotext[1:]
            # . 제외 나머지
            
            if ext.lower() not in extensions:
                continue

            filepath = os.path.join(image_dir, category, f)
            image_list.append([filepath, label])

    return image_list


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.

    # 이미지 크기 조절
    # 이미지 width, 이미지 height
    if img.shape != (img_h, img_w):
        img = cv2.resize(img, (img_w, img_h))

    return img.reshape(img_h, img_w, img_c)


def batch(paths, idx, batch_size):
    num_imgs = len(paths)
    idx1 = idx * batch_size
    idx2 = (idx + 1) * batch_size
    if idx2 > num_imgs:
        idx2 = num_imgs
    batch_list = paths[idx1: idx2]

    imgs, labels = [], []
    for i in range(batch_size):
        imgs.append(read_image(batch_list[i][0]))
        labels.append(batch_list[i][1])
    return imgs, labels


def main(argv):
    train_list = create_image_list(train_img_dir)
    if train_list is None:
        return

    print('Train image nums :', len(train_list))

    X = tf.placeholder(tf.float32, [None, img_h, img_w, img_c], name='data')
    Y = tf.placeholder(tf.int32, [None])

    conv1 = tf.layers.conv2d(X, 64, [3, 3], padding='same', activation=tf.nn.relu)
    conv1_1 = tf.layers.conv2d(conv1, 64, [3, 3], padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1_1, [2, 2], strides=2)

    conv2 = tf.layers.conv2d(pool1, 128, [3, 3], padding='same', activation=tf.nn.relu)
    conv2_1 = tf.layers.conv2d(conv2, 128, [3, 3], padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2_1, [2, 2], strides=2)

    conv3 = tf.layers.conv2d(pool2, 256, [3, 3], padding='same', activation=tf.nn.relu)
    conv3_1 = tf.layers.conv2d(conv3, 256, [3, 3], padding='same', activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv3, [2, 2], strides=2)

    conv4 = tf.layers.conv2d(pool2, 12, [3, 3], activation=tf.nn.relu)

    flat1 = tf.layers.flatten(conv4)
    dense1 = tf.layers.dense(flat1, 512, activation=tf.nn.relu)
    dense2 = tf.layers.dense(flat1, 512, activation=tf.nn.relu)

    logits = tf.layers.dense(dense2, num_cls, activation=None)
    final_tensor = tf.nn.softmax(logits, name='prob')

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits))
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost)

    # 훈련 시작
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        total_batch = int(len(train_list) / FLAGS.batch_size)
        print('Batch count:', total_batch)

        print('Learning start')
        for epoch in range(FLAGS.num_epochs):
            
            total_cost = 0
            np.random.shuffle(train_list)

            for i in range(total_batch):
                imgs, labels = batch(train_list, i, FLAGS.batch_size)
                _, cost_val = sess.run([optimizer, cost], feed_dict={X: imgs, Y: labels})
                total_cost += cost_val

            print('Epoch: %d, cost: %.8f' % ((epoch + 1), total_cost))

            if (total_cost < FLAGS.minimum_cost):
                print('Learning stoped because cost <', FLAGS.minimum_cost)
                break

        print('Learning finish')

        output_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['prob'])
        with gfile.FastGFile(FLAGS.output_model, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

    input('Press Enter to continue')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=200,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=30,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--minimum_cost',
        type=float,
        default=0.000001,
        help='Minimum cost to stop learning.'
    )
    parser.add_argument(
        '--output_model',
        type=str,
        default='./face_rec.pb',
        help='Output model file name.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
