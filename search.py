import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torchvision.models as models
import torchvision.transforms as transforms
import sys
sys.path.append("..") 
from dataset import ImagetNetTrainDataset,ImagetNetTestDataset
from Normalize import Normalize
from torchvision.datasets import ImageFolder
import warnings
from per_quilt_pert_pgd import universal_perturbation
warnings.filterwarnings("ignore")
import numpy as np
from torch.utils.data import DataLoader
import torch
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# data path
training_data_path = '/mnt/ssd/yihao/Common_dataset/ILSVRC2012_img_train/'
testing_data_path = '/mnt/ssd/yihao/Common_dataset/ILSVRC2012_img_val/'
synet_path = '/mnt/ssd/yihao/Common_dataset/imagenet2012_validation_labels.txt'
json_path = '/mnt/ssd/yihao/Common_dataset/imagenet_class_index.json'

# select number of images per class for training
classes_number = 1000
image_per_class = 1
test_image_number = 50000
# attack strength 
epsilon = 10 / 255.0
print(epsilon)

# untarget attack and target attack
# target_list = [84,101,102,320,324,327,368,385,404,409,421,427,446,456,483,487,545,562,596,629,633,701,723,745,749,777,836,954,963,971,985,987]
target_list = [None]

# target model selection
# model_name_list=["resnet50", "vgg19", "densenet", "mobilenet", "googlenet", "alexnet", "inception"]
model_name_list=["resnet50"]

# loss selection
# loss_name_list = ["SGD_UAP"]
loss_name_list = ["SGD_UAP"]

# cut ratio selection
# cut_ratio_list = [1,2,4,8,16,32]
cut_ratio_list = [2]

# epochs
epoch = 20

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

print('loader data')
train_dataset = ImagetNetTrainDataset(training_data_path, classes_number, image_per_class, transforms = transform)
val_dataset = ImagetNetTestDataset(testing_data_path, test_image_number, transforms = transform)

print('Computation')
start_time = time.time()
for model_name in model_name_list:
    for target_label in target_list:
        for loss_name in loss_name_list:
            for cut_ratio in cut_ratio_list:
                # save path
                perturbation_save_folder = '/mnt/ssd/yihao/Universal_adv_attack_total/Pytorch_UAP/perturbation_result_test/per_pgd_quilt/target_tarfoolrate_metric_'+str(target_label)+'_stop_eps_largeval_'+str(epsilon)+'/perturbation_result_UAP_PGD_modelname_'+str(model_name)+'_trainset_'+str(len(train_dataset))+'_class_number_'+str(classes_number)+'_image_per_class_'+str(image_per_class)+'_quilt_ratio_'+str(cut_ratio)+"_long_epoch/"
                v = universal_perturbation(train_dataset, val_dataset, testing_data_path, synet_path, json_path, perturbation_save_folder, model_name, cut_ratio, epoch, epsilon, beta = 12, step_decay = 0.8, batchsize=100, classes_number = classes_number,image_per_class=image_per_class, loss_name = loss_name, y_target = target_label) 
            end_time = time.time()
            print(loss_name, "time:", end_time-start_time)






