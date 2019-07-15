#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/6/5
@Description:
"""
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_log(log_file_path):
    data = pd.read_csv(log_file_path)
    train_loss = data['loss'].tolist()
    val_loss = data['val_loss'].tolist()
    train_acc = data['acc'].tolist()
    val_acc = data['val_acc'].tolist()
    max_acc_index = val_acc.index(max(val_acc))
    # print("Train Loss", train_loss)
    # print("Val Loss", val_loss)
    print(data.iloc[max_acc_index])

    plt.rcParams['figure.figsize'] = (15, 6)
    plt.figure("Result of Training and Validation")
    plt.subplot(1, 2, 1)
    plt.xlabel("step", fontsize=14)
    plt.ylabel("loss", fontsize=15)
    plt.plot(train_loss, color='green', label='train loss')
    plt.plot(val_loss, color='red', label='val loss')
    plt.legend()
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.xlabel("step", fontsize=14)
    plt.ylabel("acc", fontsize=15)
    plt.plot(train_acc, color='green', label='train acc')
    plt.plot(val_acc, color='red', label='val acc')
    plt.legend()
    plt.tight_layout()

    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)

    # ax = plt.gca()  # 获取到当前坐标轴信息
    # ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面

    plt.tight_layout()      # 自动调整子图参数，使之填充整个图像区域
    plt.show()


def plot_emotion_matrix(dataset, model_name, truth, prediction, accuracy):
    np.set_printoptions(precision=2)
    if dataset == 'RAF':
        class_names = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
    elif dataset == 'FER2013':
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    else:
        raise Exception("RAF or FER2013 only!")
    matrix = confusion_matrix(truth, prediction)
    plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                          title=model_name + ' ' + dataset + ' Confusion Matrix (Accuracy: %0.2f%%)' % (accuracy * 100))


def plot_progress(no, count):
    sign1 = str(no) + '/' + str(count)
    ratio = no/float(count)
    sign2 = ''
    for i in range(3*round(ratio*10)+1):
        sign2 += '='
    sign2 += '>  '
    for i in range(3*round((1-ratio)*10)):
        sign2 += ' '
    sign2 += str(round(ratio*100, 1)) + '%'
    print("%11s %s" % (sign1, sign2))

    # import time
    # from tqdm import tqdm
    #
    # # 一共200个，每次更新10，一共更新20次
    # with tqdm(total=200) as pbar:
    #     for i in range(20):
    #         pbar.update(10)
    #         time.sleep(0.1)

