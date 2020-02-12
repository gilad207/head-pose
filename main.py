import datasets
import models
import tensorflow as tf
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', help='input image directory', required=True)
args = parser.parse_args()

project_dir = os.getcwd() + '\\'

model_file = project_dir + 'model.h5'
train_dir = 'Train/'
test_dir = project_dir + args.test_dir
result_dir = project_dir + 'Result/'

train_file = 'train_list.txt'
test_file = 'test_list.txt'
result_file = 'result.txt'

input_size = 64
batch_size=20
max_epochs=25
class_num = 66

if not os.path.exists(result_dir):
        os.makedirs(result_dir)

if not os.path.exists(train_dir):
        os.makedirs(train_dir)

dataset = datasets.Dataset(train_dir, train_file, test_dir, test_file, result_dir, result_file, batch_size=batch_size, input_size=input_size)
net = models.ResNet50(dataset, class_num)
net.train(model_file, max_epoches=max_epochs, load_weight=True, should_train=False)
net.test()