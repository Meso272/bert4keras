#! -*- coding:utf-8 -*-
# 情感分析例子，加载albert_zh权重(https://github.com/brightmart/albert_zh)

import json
import numpy as np
import codecs
from bert4keras.backend import set_gelu
from bert4keras.tokenizer import Tokenizer
from bert4keras.bert import build_bert_model
from bert4keras.train import PiecewiseLinearLearningRate
from bert4keras.snippets import sequence_padding
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback
from bert4keras.tokenizer import SpTokenizer

from keras.utils import to_categorical
set_gelu('tanh') # 切换gelu版本


maxlen = 256
config_path = 'models/albert_base/albert_config.json'
checkpoint_path = 'models/albert_base/variables/variables'
spm_path = 'models/albert_base/assets/30k-clean.model'


def load_data(filename):
    D = []
    with codecs.open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data('datasets/IMDB_trainshuffle.data')
valid_data = load_data('datasets/IMDB_valshuffle.data')
test_data = load_data('datasets/IMDB_testshuffle.data')

# 建立分词器
tokenizer = SpTokenizer(spm_path)
albert = build_bert_model(config_path, checkpoint_path, with_pool=True,albert=True,return_keras_model=False)

class data_generator:
    """数据生成器
    """
    def __init__(self, data, batch_size=128):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            text, label = self.data[i]
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                #batch_labels = to_categorical(sequence_padding(batch_labels))
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d


# 加载预训练模型
'''
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    albert=True,
    return_keras_model=False,
)
'''
output = Dropout(rate=0.1)(albert.model.output)
#output=Flatten(input_shape=(768,))(output)
output = Dense(units=2,
               activation='softmax',
               kernel_initializer=albert.initializer)(output)

model = Model(albert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=PiecewiseLinearLearningRate(Adam(1e-4), {1000: 1, 2000: 0.1}),
    metrics=['accuracy'],
)


# 转换数据集
train_generator = data_generator(train_data)
valid_generator = data_generator(valid_data)
test_generator = data_generator(test_data)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(Callback):
    def __init__(self):
        self.best_val_acc = 0.
    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(u'val_acc: %05f, best_val_acc: %05f, test_acc: %05f\n'
              % (val_acc, self.best_val_acc, test_acc))


evaluator = Evaluator()
model.fit_generator(train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=10,
                    callbacks=[evaluator])

model.load_weights('best_model.weights')
print(u'final test acc: %05f\n' % (evaluate(test_generator)))
