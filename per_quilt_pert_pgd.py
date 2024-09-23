import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
import cv2
from tqdm import tqdm
from torch.autograd import Variable
import torchvision.models as models
import json
import copy
from Normalize import Normalize
import torch.nn.functional as F
from collections import Counter

def get_image_label_of_ImageNet(data_dir, synet_path, json_path):
    synet_idx = []
    map_class_id = {}
    label_of_image_name = {}

    labels = []
    image_names = []

    with open(synet_path, 'r') as f:
        for num, line in enumerate(f.readlines()):
            synet_idx.append((line.strip()).replace('\n', ''))

    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
        for key, value in load_dict.items():
            map_class_id[value[0]] = key

    files = sorted(os.listdir(data_dir))
    for idx, img_file in enumerate(files):
        img_num = (img_file.split('_')[2]).split('.')[0]
        for i in range(8):
            if img_num[i] != '0':
                pair = img_num[i:]
                line_id = int(pair)-1
                class_name = synet_idx[line_id]
                model_id = map_class_id[class_name]

                label_of_image_name[img_file] = model_id
                break

    return label_of_image_name

def clamped_loss(output, target, beta):
    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    loss = torch.mean(torch.min(loss_fn(output, target), beta))
    return loss

def show_run_time_UAP(UAP,path):
    image_format_perturbation = UAP.cpu().numpy()
    image_format_perturbation = np.transpose(image_format_perturbation,(1,2,0))
    image_format_perturbation = (image_format_perturbation*255.0).astype(np.uint8)
    image_format_perturbation = np.clip(image_format_perturbation, 0, 255)
    image_format_perturbation = cv2.cvtColor(image_format_perturbation, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_format_perturbation)

def universal_perturbation(train_dataset,
                           val_dataset,
                           testing_data_path,
                           synet_path,
                           json_path,
                           perturbation_save_folder,
                           model_name,
                           cut_ratio = 1,
                           nb_epoch=10, 
                           eps=10/255.0, 
                           beta = 12, 
                           step_decay = 0.8,
                           batchsize=1, 
                           classes_number=1000,
                           image_per_class =1,
                           loss_name = "SGD_UAP",
                           y_target = None, 
                           loss_fn = None, 
                           layer_name = None, 
                           uap_init = None):
    """
    """
    print(cut_ratio)
    batch_ratio = 100/batchsize
    best_fooling = 0
    n_images_mean = batchsize
    num_images = len(val_dataset)   # The length of testing data
    label_of_image_name = get_image_label_of_ImageNet(
        testing_data_path,synet_path,json_path)

    if model_name=="resnet50":
        target_model = models.resnet50(pretrained=True)
        # target_model = timm.create_model('resnet50', pretrained=True)
        extract_list = ["fc"]
    if model_name=="resnet152":
        target_model = models.resnet152(pretrained=True)
    if model_name=="vgg16":
        target_model = models.vgg16(pretrained=True)
    if model_name=="vgg19":
        target_model = models.vgg19(pretrained=True)
        extract_list = ["classifier"]
    if model_name=="densenet":
        target_model = models.densenet121(pretrained=True)
        extract_list = ["fc"]
    if model_name=="mobilenet":
        target_model = models.mobilenet_v2(pretrained=True)
        extract_list = ["fc"]
    if model_name=="alexnet":
        target_model = models.alexnet(pretrained=True)
    if model_name=="googlenet":
        target_model = models.googlenet(pretrained=True)
    if model_name=="inception":
        target_model = models.inception_v3(pretrained=True)
    
    # the model need normalize before
    net = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), target_model).eval().cuda()

    # data preparation
    train_dataloader = DataLoader(train_dataset, batch_size = n_images_mean, num_workers = 2, shuffle = True, pin_memory=True)
    _,x_train = next(enumerate(train_dataloader))

    # input parameters, the UAP is delta, but the batch_delta is used for perturbe the original images, 
    # it is inferenced by delta and can achieve the grad
    # batch_delta = torch.zeros_like(x_train).cuda() # initialize as zero vector
    delta = torch.zeros((3,224,224)).cuda()
    patch_delta = delta[:,0:int(224/cut_ratio),0:int(224/cut_ratio)].unsqueeze(0)
    losses = []
    
    # loss function
    beta = torch.cuda.FloatTensor([beta])         
    # batch_delta.requires_grad_()
    delta.requires_grad_()
    patch_delta.requires_grad_()

    for epoch in range(nb_epoch):
        print('epoch %i/%i' % (epoch + 1, nb_epoch))
        
        # perturbation step size with decay
        eps_step = eps * step_decay
        
        for i,x_train in enumerate(tqdm(train_dataloader)):
            x_train = x_train.cuda()
            ori_outputs = net(x_train)
            y_train = torch.argmax(ori_outputs,dim=1)
            # batch_delta.grad.data.zero_()
            patch_delta.requires_grad_()

            delta.data = patch_delta.repeat([x_train.shape[0], 1, 1, 1])
            batch_delta = torch.tile(delta,(1,1,cut_ratio, cut_ratio))

            # for targeted UAP, switch output labels to y_target
            if y_target is not None: y_train = torch.ones(size = y_train.shape, dtype = y_train.dtype) * y_target
            y_train = y_train.cuda()

            perturbed = torch.clamp((x_train + batch_delta).cuda(), 0, 1)
            outputs = net(perturbed)

            #SGD-UAP loss
            SGD_UAP_loss = clamped_loss(outputs, y_train.cuda(),beta)
            y_train = F.one_hot(y_train,num_classes=1000).float()

            loss = SGD_UAP_loss

            if y_target is not None: loss = -loss # minimize loss for targeted UAP
            losses.append(torch.mean(loss))
            loss.backward()

            if i % batch_ratio == 0:
                grad_sign = delta.grad.data.mean(dim = 0).sign()
                patch_delta = patch_delta + grad_sign * eps_step
                patch_delta = torch.clamp(patch_delta, -eps, eps)
                delta.grad.data.zero_()

    perturbation = patch_delta.data
    # perturbation = perturbation.unsqueeze(0)
    perturbation = torch.tile(perturbation,(1,1,cut_ratio, cut_ratio))
    # Perturb the dataset with computed perturbation
    # dataset_perturbed = dataset + v
    est_labels_orig = torch.zeros((num_images)).cuda()
    est_labels_pert = torch.zeros((num_images)).cuda()
    est_labels_real = torch.zeros((num_images)).cuda()

    tmp_batch_size = 50
    test_dataloader = DataLoader(val_dataset, batch_size = tmp_batch_size, num_workers = 2, shuffle = False, pin_memory=True)
    # Compute the estimated labels in batches
    ii = 0
    with torch.no_grad():
        for img_batch,img_name_list in tqdm(test_dataloader):
            m = (ii * tmp_batch_size)
            M = min((ii + 1) * tmp_batch_size, num_images)
            img_batch = img_batch.cuda()

            per_img_batch = Variable(img_batch + torch.tensor(perturbation).cuda()).cuda()
            ii += 1
            # print(img_batch.shape)
            # print(m, M)
            real_label_list = []
            for img_name in img_name_list:
                real_label_list.append(int(label_of_image_name[img_name]))
            est_labels_orig[m:M] = torch.argmax(net(img_batch), dim=1)
            est_labels_pert[m:M] = torch.argmax(net(per_img_batch), dim=1)
            est_labels_real[m:M] = torch.tensor(real_label_list)

        # Compute the fooling rate
        fooling_rate = torch.sum(est_labels_pert != est_labels_orig).float() / num_images
        # print(torch.sum(est_labels_pert != est_labels_orig).float())
        print('FOOLING RATE = ', fooling_rate)
        
        accuracy_rate = torch.sum(est_labels_real == est_labels_orig).float() / num_images
        # print(torch.sum(est_labels_real == est_labels_orig).float())
        print('ACCURACY RATE = ', accuracy_rate)

        per_accuracy_rate = torch.sum(est_labels_real == est_labels_pert).float() / num_images
        # print(torch.sum(est_labels_real == est_labels_pert).float())
        print('PER ACCURACY RATE = ', per_accuracy_rate)

    save_perturbation = perturbation.squeeze().cpu().numpy()
    folder = os.path.exists(perturbation_save_folder)
    if not folder:                   
        os.makedirs(perturbation_save_folder)

    pertbation_name = 'PGD-target-quilt-'+model_name+'_y_target_'+str(y_target)+'-trainset-{:.1f}-testset-{:.1f}-batchsize-{:.1f}-numepo-{:.1f}-epsilon-{:.3f}-perturbation-{:.2f}-foolrate-{:.2f}'.format(len(train_dataset),len(val_dataset), n_images_mean, nb_epoch,eps,abs(save_perturbation).max(), fooling_rate*100)
    np.save(perturbation_save_folder+pertbation_name+".npy", save_perturbation)

    # print(save_perturbation.shape)
    image_format_perturbation = save_perturbation
    image_format_perturbation = np.transpose(image_format_perturbation,(1,2,0))
    image_format_perturbation = (image_format_perturbation*255.0).astype(np.uint8)
    image_format_perturbation = np.clip(image_format_perturbation, 0, 255)
    image_format_perturbation = cv2.cvtColor(image_format_perturbation, cv2.COLOR_RGB2BGR)
    # Save to certain path
    save_img_name = 'PGD-target-quilt-'+model_name+'_y_target_'+str(y_target)+'-trainset-{:.1f}-testset-{:.1f}-batchsize-{:.1f}-numepo-{:.1f}-epsilon-{:.3f}-perturbation-{:.2f}-foolrate-{:.2f}'.format(len(train_dataset),len(val_dataset),n_images_mean,nb_epoch,eps,abs(save_perturbation).max(), fooling_rate*100)
    cv2.imwrite(perturbation_save_folder+save_img_name+".jpg", image_format_perturbation)

    big_image_format_perturbation = save_perturbation
    big_image_format_perturbation = np.transpose(big_image_format_perturbation,(1,2,0))
    big_image_format_perturbation = (big_image_format_perturbation - np.min(big_image_format_perturbation)) / (np.max(big_image_format_perturbation) - np.min(big_image_format_perturbation))
    # print(big_image_format_perturbation)
    big_image_format_perturbation = (big_image_format_perturbation*255.0).astype(np.uint8)
    big_image_format_perturbation = np.clip(big_image_format_perturbation, 0, 255)
    big_image_format_perturbation = cv2.cvtColor(big_image_format_perturbation, cv2.COLOR_RGB2BGR)
    # Save to certain path
    save_img_name = 'PGD-enlarge-target-quilt-'+model_name+'_y_target_'+str(y_target)+'-trainset-{:.1f}-testset-{:.1f}-batchsize-{:.1f}-numepo-{:.1f}-epsilon-{:.3f}-perturbation-{:.2f}-foolrate-{:.2f}'.format(len(train_dataset),len(val_dataset),n_images_mean,nb_epoch,eps,abs(save_perturbation).max(), fooling_rate*100)
    cv2.imwrite(perturbation_save_folder+save_img_name+".jpg", big_image_format_perturbation)

    print("cut ratio:",cut_ratio)
    return perturbation

    