import os
import json
import glob
import shutil
import numpy as np
import SimpleITK as sitk
from pprint import pprint


class Dataset_info:
    def __init__(self):
        self.dataset_dir = "/home/jianglei/VCL-Project/data/2022Jianglei/dataset/ImageCHD_split_sdf"
        self.train_images = self.get_train_images()
        self.train_labels = self.get_train_labels()
        self.test_images = self.get_test_images()
        self.test_labels = self.get_test_labels()
    
    def get_train_images(self):
        return sorted(glob.glob(f'{self.dataset_dir}/train/images/*image.nii.gz'))
    def get_train_labels(self):
        return sorted(glob.glob(f'{self.dataset_dir}/train/labels/*label.nii.gz'))
    def get_test_images(self):
        return sorted(glob.glob(f'{self.dataset_dir}/test/images/*image.nii.gz'))
    def get_test_labels(self):
        return sorted(glob.glob(f'{self.dataset_dir}/test/labels/*label.nii.gz'))
    

    """
    用于查看ImageCHD原始数据集:数据根路径、数据规模、图像和掩码图像路径
    """
    def print_dataset_info(self):
        print(f"数据集路径：{self.dataset_dir}")
        print(f"train images of len: {len(self.train_images)}")
        print(f"train labels of len: {len(self.train_labels)}")
        print(f"test images of len: {len(self.test_images)}")
        print(f"test labels of len: {len(self.test_labels)}")

    def print_dataset(self):
        self.print_dataset_train()
        self.print_dataset_test()

    def print_dataset_train(self):
        for label_path in self.train_labels:
            label = sitk.ReadImage(label_path)
            size = label.GetSize()
            spacing = label.GetSpacing()
            np_data = sitk.GetArrayFromImage(label)
            print("Label path: ", label_path)
            print("Size:", size)
            print("Space:", spacing)
            print("Pixel value range:", np.unique(np_data))
            
    def print_dataset_test(self):
        for label_path in self.test_labels:
            label = sitk.ReadImage(label_path)
            size = label.GetSize()
            spacing = label.GetSpacing()
            np_data = sitk.GetArrayFromImage(label)
            print("Label path: ", label_path)
            print("Size:", size)
            print("Space:", spacing)
            print("Pixel value range:", np.unique(np_data))

    """
    根据dataset_split_3D.json中对train和test的数据划分
    数据划分json: dataset/ImageCHD_processed/dataset_split_3D.json
    原始数据路径: dataset/ImageCHD
    输出数据路径: dataset/ImageCHD_split
    """
    def split_dataset(self):
        json_path = "/home/jianglei/VCL-Project/data/2022Jianglei/dataset/ImageCHD_processed/dataset_split_3D.json"
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for item in data["training"]:
            image_path = item["image"]
            label_path = item["label"]
            new_dir = "/home/jianglei/VCL-Project/data/2022Jianglei/dataset/ImageCHD_split/train"
            new_image_path = os.path.join(new_dir, "images", image_path.split("/")[-1])
            new_label_path = os.path.join(new_dir, "labels", label_path.split("/")[-1])
            shutil.copy(image_path, new_image_path)
            shutil.copy(label_path, new_label_path)
        for item in data["test"]:
            image_path = item["image"]
            label_path = item["label"]
            new_dir = "/home/jianglei/VCL-Project/data/2022Jianglei/dataset/ImageCHD_split/test"
            new_image_path = os.path.join(new_dir, "images", image_path.split("/")[-1])
            new_label_path = os.path.join(new_dir, "labels", label_path.split("/")[-1])
            shutil.copy(image_path, new_image_path)
            shutil.copy(label_path, new_label_path)

    """
    根据论文SDF4CHD进行划分, 划分为train、val、test
    原始数据路径: dataset/ImageCHD
    输出数据路径: dataset/ImageCHD_split_sdf
    """
    def split_dataset_v2(self):
        train_ids = {1001,1007,1008,1012,1013,1015,1017,1018,1021,1022,1024,1025,1029,1032,1037,1041,1051,1052,1053,1056,1061,1062,1066,1067,1072,1074,1075,1077,1078,1079,
                    1080,1081,1085,1088,1098,1099,1101,1102,1103,1106,1109,1110,1111,1113,1116,1117,1120,1121,1122,1124,1125,1126,1127,1128,1132,1133,1139,1140,1141,1144,1145,1146,1147,
                    1148,1158,1161,1170,1178}
        val_ids = {1019,1020,1039,1042,1047,1091,1129,1143}
        test_ids = {1002,1003,1004,1005,1010,1011,1014,1016,1023,1028,1030,1033,1035,1036,1043,1044,1046,1048,1050, 1054,1059,1060,1063,1064,1070,1083,1092,1105,1112,1114,
                    1119,1135,1138,1150}
        print(len(train_ids))
        print(len(val_ids))
        print(len(test_ids))
        raw_dir = "/home/jianglei/VCL-Project/data/2022Jianglei/dataset/ImageCHD"
        new_dir = "/home/jianglei/VCL-Project/data/2022Jianglei/dataset/ImageCHD_split_sdf"
        image_filename = "ct_id_image.nii.gz"
        label_filename = "ct_id_label.nii.gz"
        # for train_id in train_ids:
        #     raw_image_filename = image_filename.replace("id", str(train_id))
        #     raw_label_filename = label_filename.replace("id", str(train_id))
        #     raw_image_path = os.path.join(raw_dir, raw_image_filename)
        #     raw_label_path = os.path.join(raw_dir, raw_label_filename)
        #     new_image_path = os.path.join(new_dir, "train", "images", raw_image_filename)
        #     new_label_path = os.path.join(new_dir, "train", "labels", raw_label_filename)
        #     shutil.copy(raw_image_path, new_image_path)
        #     shutil.copy(raw_label_path, new_label_path)

        for val_id in val_ids:
            raw_image_filename = image_filename.replace("id", str(val_id))
            raw_label_filename = label_filename.replace("id", str(val_id))
            raw_image_path = os.path.join(raw_dir, raw_image_filename)
            raw_label_path = os.path.join(raw_dir, raw_label_filename)
            new_image_path = os.path.join(new_dir, "val", "images", raw_image_filename)
            new_label_path = os.path.join(new_dir, "val", "labels", raw_label_filename)
            shutil.copy(raw_image_path, new_image_path)
            shutil.copy(raw_label_path, new_label_path)

        for test_id in test_ids:
            raw_image_filename = image_filename.replace("id", str(test_id))
            raw_label_filename = label_filename.replace("id", str(test_id))
            raw_image_path = os.path.join(raw_dir, raw_image_filename)
            raw_label_path = os.path.join(raw_dir, raw_label_filename)
            new_image_path = os.path.join(new_dir, "test", "images", raw_image_filename)
            new_label_path = os.path.join(new_dir, "test", "labels", raw_label_filename)
            shutil.copy(raw_image_path, new_image_path)
            shutil.copy(raw_label_path, new_label_path)

if __name__ == "__main__":
    dataset = Dataset_info()
    dataset.print_dataset_test()