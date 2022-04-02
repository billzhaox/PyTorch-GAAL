import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.nn import functional as nF
import torchvision.datasets as dset
from torch.backends import cudnn

from utils import parse_args, plot, plot_err, plot_all, generate_run_id
from gan.dcgan import Generator
from oracle.cnn import CNN
from oracle.dla_simple import SimpleDLA
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn import svm
from oracle import cnn
import random
from sklearn.metrics import accuracy_score

from torchmin import minimize


# random.seed(1)

class TrainLoop:
    def __init__(self, args, device, init_labeled_data, init_labeled_targets, test_data, test_targets, id, rt=999):

        self.id = id
        self.rt = rt
        self.args = args
        self.device = device
        self.labeled_data_rec = init_labeled_data
        self.labeled_targets_rec = init_labeled_targets
        self.labeled_data = init_labeled_data
        self.labeled_targets = init_labeled_targets
        self.test_data = test_data
        self.test_targets = test_targets
        self.acc_hist_gaal = []
        self.acc_hist_random = []

        _, self.svm_W_tensor, self.svm_b = self.train_svm(init_labeled_data, init_labeled_targets)

        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        if args.dset == 'mnist57' or args.dset == 'USPS':
            # Load generator G
            ngpu = 1
            self.gen = Generator(ngpu, mode=0)
            self.gen.load_state_dict(torch.load("./gan/generator/G_MNIST.pth".format(device_type)))
            # self.gen = torch.nn.DataParallel(self.gen).to(self.device)
            self.gen = self.gen.to(self.device)
            self.gen.eval()

            # Load oracle classifier
            self.ora = CNN()
            device_type = "GPU" if torch.cuda.is_available() else "CPU"
            self.ora.load_state_dict(torch.load("./oracle/CNN_mnist57.pth".format(device_type)))
            # self.ora = torch.nn.DataParallel(self.ora)
            self.ora.eval()
        elif args.dset == 'CIFAR10':
            # Load generator G
            ngpu = 1
            self.gen = Generator(ngpu, mode=1).to(self.device)
            self.gen.load_state_dict(torch.load("./gan/generator/G_CIFAR.pth".format(device_type)))
            self.gen.eval()

            # Load oracle classifier
            self.ora = SimpleDLA().to(self.device)
            if device_type == "GPU":
                self.ora = torch.nn.DataParallel(self.ora)
                cudnn.benchmark = True
            checkpoint = torch.load('./oracle/ORA_CIFAR.pth')
            self.ora.load_state_dict(checkpoint['net'])
            self.ora.eval()

    def train_svm(self, x, y):
        clf = svm.SVC(kernel='linear', gamma=0.001)
        clf.fit(x, y)
        W = clf.coef_[0]  # W consists of 28*28=784 elements
        b = clf.intercept_[0]  # b consists of 1 element
        W_tensor = torch.from_numpy(W).float().to(self.device)
        W_tensor.requires_grad_(False)
        pred_labels = clf.predict(self.test_data)
        # print(pred_labels)
        acc = accuracy_score(self.test_targets, pred_labels)
        # plt.plot(numlabel_hist,acc_hist,'o-')
        # plt.show()
        return acc, W_tensor, b

    # loss function -- uncertainty
    def loss_op(self, z):
        fake = self.gen(z)
        # print(fake.shape)  # 1*1*28*28
        fake_res = torch.flatten(torch.squeeze(fake))  # G(z)
        loss = torch.dot(self.svm_W_tensor, fake_res) + self.svm_b  # W * G(z) + b
        return torch.norm(loss, p=2, dim=0)
        # return torch.abs(loss)

    # loss function -- uncertainty + diversity (L2 distance with the current batch, maximize the min)
    def loss_op_diver1_1(self, z):
        fake = self.gen(z)
        # print(fake.shape)  # 1*1*28*28
        fake_res = torch.flatten(fake)  # G(z)
        fake_flatten = torch.flatten(fake, start_dim=1)
        loss = torch.dot(self.svm_W_tensor, fake_res) + self.svm_b  # W * G(z) + b
        org = torch.norm(loss, p=2, dim=0)
        if self.cnt == 0:
            return org
        else:
            diver = torch.cdist(fake_flatten, self.gen_list, p=2).min()
            return org - 0.001 * diver
        # return torch.abs(loss)

    # loss function -- uncertainty + diversity (L2 distance with all labelled data, maximize the min)
    def loss_op_diver1_2(self, z):
        fake = self.gen(z)
        # print(fake.shape)  # 1*1*28*28
        fake_res = torch.flatten(fake)  # G(z)
        fake_flatten = torch.flatten(fake, start_dim=1)
        loss = torch.dot(self.svm_W_tensor, fake_res) + self.svm_b  # W * G(z) + b
        org = torch.norm(loss, p=2, dim=0)
        diver = torch.cdist(fake_flatten, self.labeled_data.to(self.device), p=2).min()
        return org - 0.001 * diver
        # return torch.abs(loss)

    # loss function -- uncertainty + diversity (Cosine Similarity with the current batch, minimize the max)
    def loss_op_diver2_1(self, z):
        fake = self.gen(z)
        # print(fake.shape)  # 1*1*28*28
        fake_res = torch.flatten(fake)  # G(z)
        fake_flatten = torch.flatten(fake, start_dim=1)
        loss = torch.dot(self.svm_W_tensor, fake_res) + self.svm_b  # W * G(z) + b
        org = torch.norm(loss, p=2, dim=0)
        if self.cnt == 0:
            return org
        else:
            diver = nF.cosine_similarity(fake_flatten, self.gen_list).max()
            return org + 0.0001 * diver
        # return torch.abs(loss)

    # loss function -- uncertainty + diversity (Cosine Similarity with all labelled data, minimize the max)
    def loss_op_diver2_2(self, z):
        fake = self.gen(z)
        # print(fake.shape)  # 1*1*28*28
        fake_res = torch.flatten(fake)  # G(z)
        fake_flatten = torch.flatten(fake, start_dim=1)
        loss = torch.dot(self.svm_W_tensor, fake_res) + self.svm_b  # W * G(z) + b
        org = torch.norm(loss, p=2, dim=0)
        diver = nF.cosine_similarity(fake_flatten, self.labeled_data.to(self.device)).max()
        return org + 0.0001 * diver
        # return torch.abs(loss)

    '''
        typ - 
            '0': loss_op
            '1_1': loss_op_diver1_1
            '1_2': loss_op_diver1_2
            '2_1': loss_op_diver2_1
            '2_2': loss_op_diver2_2
    '''
    def train_gaal(self,typ='0'):
        self.acc_hist_gaal = []
        self.labeled_data = self.labeled_data_rec
        self.labeled_targets = self.labeled_targets_rec
        limit = self.args.limit
        num_label = 50
        # init linear SVM
        ac, self.svm_W_tensor, self.svm_b = self.train_svm(self.labeled_data, self.labeled_targets)
        self.acc_hist_gaal.append(ac)
        while num_label < limit:

            print('Now label {}/{}'.format(num_label, limit))

            # Solve optimization problem by GD, 10 queries every time
            z_list = []
            org_loss_list = []
            loss_list = []
            mini_list = []
            iterr = 200
            self.cnt = 0
            start = time.time()
            while self.cnt < 10:
                # history = []
                # history_lr = []
                zz = torch.randn(1, 100, 1, 1, device=self.device, requires_grad=True)
                if typ == '0':
                    result = minimize(self.loss_op, zz, method='l-bfgs')
                    if result.success:
                        org_loss_list.append(float(self.loss_op(zz)))
                        mini_list.append(float(result.fun))
                        z_list.append(result.x)
                        self.cnt += 1
                        # print("Add diver", self.cnt)
                elif typ == '1_2' or typ == '2_2':
                    loss_func = self.loss_op_diver1_2 if typ == '1_2' else self.loss_op_diver2_2
                    result = minimize(loss_func, zz, method='l-bfgs')
                    if result.success:
                        org_loss_list.append(float(loss_func(zz)))
                        mini_list.append(float(result.fun))
                        z_list.append(result.x)
                        s1 = self.gen(result.x)
                        s_flatten = torch.flatten(s1, start_dim=1)
                        out = self.ora(s1.detach().cpu())
                        label = out.data.max(1)[1]
                        self.labeled_data = torch.cat((self.labeled_data, s_flatten.detach().cpu()))
                        self.labeled_targets = torch.cat((self.labeled_targets, torch.tensor([label])))
                        self.cnt += 1
                        # print("Add diver", self.cnt)
                elif typ == '1_1' or typ == '2_1':
                    loss_func = self.loss_op_diver1_1 if typ == '1_1' else self.loss_op_diver2_1
                    result = minimize(loss_func, zz, method='l-bfgs')
                    if result.success:
                        org_loss_list.append(float(loss_func(zz)))
                        mini_list.append(float(result.fun))
                        z_list.append(result.x)
                        s1 = self.gen(result.x)
                        s_flatten = torch.flatten(s1, start_dim=1)
                        # zz_flatten = torch.flatten(self.gen(zz), start_dim=1)
                        # print("mini",float(result.fun))
                        # print("org", float(self.loss_op(zz)))
                        if self.cnt == 0:
                            self.gen_list = s_flatten
                        else:
                            # print("diver", torch.cdist(s_flatten, self.gen_list, p=2))
                            # print("diver_min", 0.01 * float(torch.cdist(s_flatten, self.gen_list, p=2).min()))
                            # print("zz_diver", torch.cdist(zz_flatten, self.gen_list, p=2))
                            # print("zz_diver_min", 0.01 * float(torch.cdist(zz_flatten, self.gen_list, p=2).min()))
                            self.gen_list = torch.cat((self.gen_list, s_flatten))
                        self.cnt += 1
                        # print("Add ", self.cnt)
                    # org_loss_list.append(float(self.loss_op(zz)))
                    # # optimizer = torch.optim.SGD([zz], lr=1., momentum=0.9)
                    # optimizer = torch.optim.Adam([zz], lr=0.1)
                    # for i in range(iterr):
                    #     loss = self.loss_op(zz)
                    #     # history.append(loss.detach().cpu().clone().numpy())
                    #     optimizer.zero_grad()
                    #     loss.backward()
                    #     optimizer.step()
                    # z_list.append(zz)
                    # loss_list.append(float(self.loss_op(zz)))
                    # cnt += 1
                    # print("Add ", cnt)


            # print(z_list)
            print("Generated 10 samples")
            # print(org_loss_list)
            # print(loss_list)
            # print(mini_list)
            end = time.time()
            print("time: ", end - start)

            # Generate samples using z_list, label them, add them to the labeled dataset
            if typ != 2:
                for z in z_list:
                    fake = self.gen(z)
                    fake_flatten = torch.flatten(fake, start_dim=1)
                    out = self.ora(fake.detach().cpu())
                    label = out.data.max(1)[1]
                    # debug
                    # print("label:",label)
                    # plt.imshow(np.transpose(torch.squeeze(fake, 0).detach().cpu(), (1, 2, 0)))
                    # plt.show()
                    self.labeled_data = torch.cat((self.labeled_data, fake_flatten.detach().cpu()))
                    self.labeled_targets = torch.cat((self.labeled_targets, torch.tensor([label])))
            # print(labeled_data.shape)
            # print(labeled_targets.shape)
            num_label += self.cnt

            # re-train the SVM, update W and b
            ac, self.svm_W_tensor, self.svm_b = self.train_svm(self.labeled_data, self.labeled_targets)
            self.acc_hist_gaal.append(ac)

        # plot(self.numlabel_hist, self.acc_hist, self.args.dset, self.args.limit, self.id, self.rt)
        return self.acc_hist_gaal

    def train_random(self, unlabeled_data, unlabeled_targets):
        self.labeled_data = self.labeled_data_rec
        self.labeled_targets = self.labeled_targets_rec
        limit = self.args.limit
        num_label = 50
        # init linear SVM
        ac, self.svm_W_tensor, self.svm_b = self.train_svm(self.labeled_data, self.labeled_targets)
        self.acc_hist_random.append(ac)
        l_env = range(unlabeled_data.shape[0])
        while num_label < limit:
            # randomly select 10 samples from the unlabeled pool, add them to the labeled dataset
            l = random.sample(l_env, 10)
            batch_data, batch_targets = unlabeled_data[l], unlabeled_targets[l]
            l_env = [i for i in l_env if i not in l]
            self.labeled_data = torch.cat((self.labeled_data, batch_data))
            self.labeled_targets = torch.cat((self.labeled_targets, batch_targets))
            num_label += 10

            # re-train the SVM, update W and b
            ac, self.svm_W_tensor, self.svm_b = self.train_svm(self.labeled_data, self.labeled_targets)
            self.acc_hist_random.append(ac)

        # plot(range(50, limit + 10, 10), self.acc_hist_random, self.args.dset, limit, self.id, self.rt)
        return self.acc_hist_random


if __name__ == '__main__':
    args = parse_args()
    # specify the device on which the model will be trained
    TRAINING_ON_GPU = torch.cuda.is_available()
    device = torch.device("cuda:0" if TRAINING_ON_GPU else "cpu")
    print('GPU available: ', TRAINING_ON_GPU)

    ################### Prepare data #################################

    dataroot = 'data/'
    if args.dset == 'mnist57' or args.dset == 'USPS':
        dataset = dset.MNIST(root=dataroot, download=True)
        idx = (dataset.targets == 5) | (dataset.targets == 7)
        dataset.data, dataset.targets = torch.flatten(dataset.data[idx], start_dim=1) / 255.0, dataset.targets[idx]
        for i in range(len(dataset.targets)):
            dataset.targets[i] = 0 if dataset.targets[i] == 5 else 1
    elif args.dset == 'CIFAR10':
        dataset = dset.CIFAR10(root=dataroot, train=True, download=True)
        idx = [i for i in range(len(dataset.targets)) if dataset.targets[i] == 1 or dataset.targets[i] == 7]
        dataset.data, dataset.targets = np.take(dataset.data, idx, 0), np.take(dataset.targets, idx, 0)
        for i in range(len(dataset.targets)):
            dataset.targets[i] = 0 if dataset.targets[i] == 1 else 1
        # print(dataset.data.shape)
        # print(dataset.data[0])
        dataset.data = np.reshape(dataset.data, (len(dataset.data), -1)) / 255.0

    if args.dset == 'mnist57':
        test_dataset = dset.MNIST(root=dataroot, train=False, download=True)
        idx = (test_dataset.targets == 5) | (test_dataset.targets == 7)
        test_data, test_targets = torch.flatten(test_dataset.data[idx], start_dim=1) / 255.0, test_dataset.targets[idx]
        for i in range(len(test_targets)):
            test_targets[i] = 0 if test_targets[i] == 5 else 1
        # print(test_dataset.data.shape)  # [1920, 784]
    elif args.dset == 'USPS':
        test_dataset2 = dset.USPS(root=dataroot, train=True, download=True)
        idx = [i for i in range(len(test_dataset2.targets)) if
               test_dataset2.targets[i] == 5 or test_dataset2.targets[i] == 7]
        test_dataset2.data, test_targets = np.take(test_dataset2.data, idx, 0), np.take(test_dataset2.targets, idx,
                                                                                        0)
        # print(test_targets)
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(4),
            transforms.Resize([28, 28])
        ])
        test_data = torch.empty(len(test_dataset2.data), 28, 28)
        for i in range(len(test_dataset2.data)):
            res = trans(test_dataset2.data[i])
            test_data[i] = res.clone()
        test_data = np.reshape(test_data, (len(test_data), -1))
        for i in range(len(test_targets)):
            test_targets[i] = 0 if test_targets[i] == 5 else 1
    elif args.dset == 'CIFAR10':
        test_dataset = dset.CIFAR10(root=dataroot, train=True, download=True)
        idx = [i for i in range(len(test_dataset.targets)) if
               test_dataset.targets[i] == 1 or test_dataset.targets[i] == 7]
        test_data, test_targets = np.take(test_dataset.data, idx, 0), np.take(test_dataset.targets, idx, 0)
        for i in range(len(test_targets)):
            test_targets[i] = 0 if test_targets[i] == 1 else 1
        test_data = np.reshape(test_data, (len(test_data), -1)) / 255.0

    ##################################################################

    # randomly select 50 data as the initial labeled dataset
    l = random.sample(range(dataset.data.shape[0]), 50)
    labeled_data, labeled_targets = dataset.data[l], dataset.targets[l]
    # print(labeled_data.shape) # [50, 784]
    l_env = [i for i in range(dataset.data.shape[0]) if i not in l]
    unlabeled_data, unlabeled_targets = dataset.data[l_env], dataset.targets[l_env]
    if args.dset == 'CIFAR10':
        labeled_data, labeled_targets = torch.from_numpy(labeled_data), torch.from_numpy(labeled_targets)
        unlabeled_data, unlabeled_targets = torch.from_numpy(unlabeled_data), torch.from_numpy(unlabeled_targets)
    # print("Type after conversion:\n", type(labeled_data))

    gaal_list = []
    gaal_list_diver1_1 = []
    gaal_list_diver1_2 = []
    gaal_list_diver2_1 = []
    gaal_list_diver2_2 = []
    random_list = []
    full_list = []
    total_numlabel = range(50, args.limit + 10, 10)
    # averaged over 10 runs
    id = generate_run_id()
    for run in range(10):
        print('Now run {}/{}'.format(run + 1, 10))
        oneTrain = TrainLoop(args, device, labeled_data, labeled_targets, test_data, test_targets, id, rt=run + 1)
        print("Training GAAL ...")
        gaal_list.append(oneTrain.train_gaal('0'))
        print("Training GAAL Diver 1_1 ...")
        gaal_list_diver1_1.append(oneTrain.train_gaal('1_1'))
        print("Training GAAL Diver 1_2 ...")
        gaal_list_diver1_2.append(oneTrain.train_gaal('1_2'))
        print("Training GAAL Diver 2_1 ...")
        gaal_list_diver2_1.append(oneTrain.train_gaal('2_1'))
        print("Training GAAL Diver 2_2 ...")
        gaal_list_diver2_2.append(oneTrain.train_gaal('2_2'))
        # print("Training Random ...")
        # random_list.append(oneTrain.train_random(unlabeled_data, unlabeled_targets))
        # print(dataset.data.shape)
        print("Training Full Supervised ...")
        full_acc, a, b = oneTrain.train_svm(dataset.data, dataset.targets)
        full_list.append(full_acc)


    plot_all(total_numlabel, gaal_list, gaal_list_diver1_1, gaal_list_diver1_2, gaal_list_diver2_1, gaal_list_diver2_2, random_list, full_list, args.dset, args.limit, id)


    # plot_err(total_numlabel, aver, sd, args.dset, args.limit, id)

    # oneTrain = TrainLoop(args, device, labeled_data, labeled_targets, test_data, test_targets, id)
    # print(labeled_data.shape)
    # print(dataset.data.shape)
    # print(test_data.shape)
    # full_acc, a, b = oneTrain.train_svm(dataset.data, dataset.targets)
    # print(dataset.data.shape)
    # print(full_acc)
