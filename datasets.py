import os
import numpy as np
import cv2
import scipy.io as sio
import utils
import face_detector

class Dataset:
    def __init__(self, train_dir, train_file, test_dir, test_file, result_dir, result_file, batch_size, input_size):
        self.train_dir = train_dir
        self.train_file = train_file
        self.test_dir = test_dir
        self.test_file = test_file
        self.result_dir = result_dir
        self.result_file = result_file
        self.batch_size = batch_size
        self.input_size = input_size
        self.train_format = '.jpg'
        self.test_format = '.png'
        self.train_num = self.__gen_filename_list(self.train_dir, self.train_file, self.train_format)
        self.test_num = self.__gen_filename_list(self.test_dir, self.test_file, self.test_format)

    def __get_ypr_from_mat(self, mat_path):
        mat = sio.loadmat(mat_path)
        pre_pose_params = mat['Pose_Para'][0]
        pose_params = pre_pose_params[:3]
        return pose_params

    def __get_pt2d_from_mat(self, mat_path):
        mat = sio.loadmat(mat_path)
        pt2d = mat['pt2d']
        return pt2d
    
    def __get_train_img(self, train_file):
        img = cv2.imread(os.path.join(self.train_dir, train_file + self.train_format))
        pt2d = self.__get_pt2d_from_mat(os.path.join(self.train_dir, train_file + '.mat'))
        crop_img = utils.crop_face(img, pt2d)
        crop_img = np.asarray(cv2.resize(crop_img, (self.input_size, self.input_size)))
        normed_img = (crop_img - crop_img.mean()) / crop_img.std()
        return normed_img

    def __get_test_img(self, test_file):
        img = cv2.imread(os.path.join(self.test_dir, test_file + self.test_format))
        pt2d = face_detector.detect_landmarks(img)
        crop_img = utils.crop_face(img, pt2d) if pt2d is not None else img
        crop_img = np.asarray(cv2.resize(img, (self.input_size, self.input_size)))
        normed_img = (crop_img - img.mean()) / crop_img.std()
        return normed_img
    
    def __get_input_label(self, file_name):
        pose = self.__get_ypr_from_mat(os.path.join(self.train_dir, file_name + '.mat'))
        
        # convert to degrees
        yaw = pose[1] * 180.0 / np.pi
        pitch = pose[0] * 180.0 / np.pi
        roll = pose[2] * 180.0 / np.pi
        
        cont_labels = [yaw, pitch, roll]
        bins = np.array(range(-99, 99, 3))
        bin_labels = np.digitize([yaw, pitch, roll], bins) - 1
        
        return bin_labels, cont_labels

    def __gen_filename_list(self, directory, file, ext):
        filename_list_file = os.path.join(directory, file)
        if os.path.exists(filename_list_file):
            os.remove(filename_list_file)
        
        count = 0
        with open(filename_list_file, 'w+') as tlf:
            for root, dirs, files in os.walk(directory):
                for f in files:
                    if os.path.splitext(f)[1] == ext:
                        tlf.write(os.path.splitext(f)[0] + '\n')
                        count = count + 1
        return count
    
    def __get_list_from_filenames(self, directory, file):
        file_path = os.path.join(directory,file)
        with open(file_path) as f:
            lines = f.read().splitlines()
        return lines

    def test_generator(self):
        filenames = self.__get_list_from_filenames(self.test_dir,self.test_file)
        batch_x = []
        names = []

        for i in range(0, self.test_num):
            img = self.__get_test_img(filenames[i])
            batch_x.append(img)
            names.append(filenames[i])

        batch_x = np.array(batch_x, dtype=np.float32)
        return batch_x, names
        
    def train_generator(self, shuffle):  
        filenames = self.__get_list_from_filenames(self.train_dir, self.train_file)
        while True:
            if shuffle:
                idx = np.random.permutation(self.train_num)
                filenames = np.array(filenames)[idx]
            max_num = self.train_num - (self.train_num % self.batch_size)
            for i in range(0, max_num, self.batch_size):
                batch_x = []
                batch_yaw = []
                batch_pitch = []
                batch_roll = []
                names = []
                for j in range(self.batch_size):
                    img = self.__get_train_img(filenames[i + j])
                    bin_labels, cont_labels = self.__get_input_label(filenames[i + j])

                    batch_x.append(img)
                    batch_yaw.append([bin_labels[0], cont_labels[0]])
                    batch_pitch.append([bin_labels[1], cont_labels[1]])
                    batch_roll.append([bin_labels[2], cont_labels[2]])
                    names.append(filenames[i + j])
                
                batch_x = np.array(batch_x, dtype=np.float32)
                batch_yaw = np.array(batch_yaw)
                batch_pitch = np.array(batch_pitch)
                batch_roll = np.array(batch_roll)
                
                yield (batch_x, [batch_yaw, batch_pitch, batch_roll])

    def save_result(self, name, yaw, pitch, roll):
        image = cv2.imread(os.path.join(self.test_dir, name + self.test_format))
        cv2_img = utils.draw_axis(image, yaw, pitch, roll, tdx=image.shape[0]//2, tdy=image.shape[1]//2,size=100)
        save_path = os.path.join(self.result_dir, name + self.test_format)
        cv2.imwrite(save_path, cv2_img)
    