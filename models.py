import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import utils
import glob
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

class ResNet50:
    def __init__(self, dataset, class_num):
        self.class_num = class_num
        self.idx_tensor = [idx for idx in range(self.class_num)]
        self.idx_tensor = tfe.Variable(np.array(self.idx_tensor, dtype=np.float32))
        self.dataset = dataset
        self.model = self.__create_model()
        
    def __loss_angle(self, y_true, y_pred, alpha=0.5):
        # cross entropy loss
        bin_true = y_true[:,0]
        cont_true = y_true[:,1]
        onehot_labels = tf.keras.utils.to_categorical(bin_true, self.class_num)
        cls_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=y_pred)
        # MSE loss
        pred_cont = tf.reduce_sum(tf.nn.softmax(y_pred) * self.idx_tensor, 1) * 3 - 99
        mse_loss = tf.losses.mean_squared_error(labels=cont_true, predictions=pred_cont)
        # Total loss
        total_loss = cls_loss + alpha * mse_loss
        return total_loss

    def __create_model(self):
        resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet',input_shape=(self.dataset.input_size,self.dataset.input_size,3))
        output = resnet.layers[-1].output
        output = tf.keras.layers.Flatten()(output) 

        fc_yaw = tf.keras.layers.Dense(name='yaw', units=self.class_num)(output)
        fc_pitch = tf.keras.layers.Dense(name='pitch', units=self.class_num)(output)
        fc_roll = tf.keras.layers.Dense(name='roll', units=self.class_num)(output)
    
        model = tf.keras.Model(inputs=resnet.inputs, outputs=[fc_yaw, fc_pitch, fc_roll])
        
        losses = {
            'yaw':self.__loss_angle,
            'pitch':self.__loss_angle,
            'roll':self.__loss_angle,
        }
        
        model.compile(optimizer=tf.train.AdamOptimizer(),loss=losses)
       
        return model

    def train(self, model_path, max_epoches, load_weight, should_train):
        
        if load_weight:
            self.model.load_weights(model_path)

        if should_train:
            self.model.fit(self.dataset.train_generator(shuffle=True),epochs=max_epoches,
                            steps_per_epoch=self.dataset.train_num // self.dataset.batch_size
                            ,max_queue_size=10,workers=1,verbose=1)

            self.model.save(model_path)
            
    def test(self):
        result_file = os.path.join(self.dataset.result_dir, self.dataset.result_file)
        if os.path.exists(result_file):
            os.remove(result_file)

        images, names = self.dataset.test_generator()
        predictions = self.model.predict(images, batch_size=len(images), verbose=1)
        predictions = np.asarray(predictions)
        yaws = tf.reduce_sum(tf.nn.softmax(predictions[0,:,:]) * self.idx_tensor, 1) * 3 - 99
        pitches = tf.reduce_sum(tf.nn.softmax(predictions[1,:,:]) * self.idx_tensor, 1) * 3 - 99
        rolls = tf.reduce_sum(tf.nn.softmax(predictions[2,:,:]) * self.idx_tensor, 1) * 3 - 99

        with open(result_file,'w+') as rf:
            for name, yaw, pitch, roll in zip(names, yaws, pitches, rolls):
                self.dataset.save_result(name, yaw, pitch, roll)
                rx, ry, rz = utils.convert_ypr_to_rvec(yaw, pitch, roll)
                rf.write(f'{name}, {rx}, {ry}, {rz}\n')
                


            
        