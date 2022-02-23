import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.datasets as dset
from utils import parse_args, plot, plot_err, plot_all, generate_run_id
from gan.dcgan import Generator
from oracle.cnn import CNN
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
        self.labeled_data_gaal = init_labeled_data
        self.labeled_targets_gaal = init_labeled_targets
        self.labeled_data_random = init_labeled_data
        self.labeled_targets_random = init_labeled_targets
        self.test_data = test_data
        self.test_targets = test_targets
        self.acc_hist_gaal = []
        self.acc_hist_random = []

        _, self.svm_W_tensor, self.svm_b = self.train_svm(init_labeled_data, init_labeled_targets)

        # Load generator G
        ngpu = 1
        self.gen = Generator(ngpu).to(self.device)
        self.gen.load_state_dict(torch.load('./gan/generator/G.pth'))
        self.gen.eval()

        # Load oracle classifier
        self.ora = CNN()
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        self.ora.load_state_dict(torch.load("./oracle/CNN_mnist57.pth".format(device_type)))

    def train_svm(self, x, y):
        clf = svm.SVC(kernel='linear', gamma=0.001)
        clf.fit(x, y)
        W = clf.coef_[0]  # W consists of 28*28=784 elements
        b = clf.intercept_[0]  # b consists of 1 element
        W_tensor = torch.from_numpy(W).float().to(self.device)
        W_tensor.requires_grad_(False)
        pred_labels = clf.predict(self.test_data)
        acc = accuracy_score(self.test_targets, pred_labels)
        # print(acc)
        # plt.plot(numlabel_hist,acc_hist,'o-')
        # plt.show()
        return acc, W_tensor, b

    # loss function we want to optimize
    def loss_op(self, z):
        fake = self.gen(z)
        # print(fake.shape)  # 1*1*28*28
        fake_res = torch.flatten(torch.squeeze(fake))  # G(z)
        loss = torch.dot(self.svm_W_tensor, fake_res) + self.svm_b  # W * G(z) + b
        return torch.abs(loss)

    def train_gaal(self):
        limit = self.args.limit
        num_label = 50
        # init linear SVM
        ac, self.svm_W_tensor, self.svm_b = self.train_svm(self.labeled_data_gaal, self.labeled_targets_gaal)
        self.acc_hist_gaal.append(ac)
        while num_label < limit:

            print('Now label {}/{}'.format(num_label, limit))

            # Solve optimization problem by GD, 10 queries every time
            z_list = []
            org_loss_list = []
            loss_list = []
            mini_list = []
            iterr = 200
            cnt = 0
            while cnt < 10:
                # history = []
                # history_lr = []
                zz = torch.randn(1, 100, 1, 1, device=self.device, requires_grad=True)
                result = minimize(self.loss_op, zz, method='l-bfgs')
                if result.success:
                    org_loss_list.append(float(self.loss_op(zz)))
                    mini_list.append(float(result.fun))
                    z_list.append(result.x)
                    cnt += 1

                # optimizer = torch.optim.SGD([zz], lr=1., momentum=0.9)
                # # optimizer = torch.optim.Adam([zz2], lr=1.)
                # for i in range(iterr):
                #     loss = loss_op(zz)
                #     history.append(loss.detach().cpu().clone().numpy())
                #     optimizer.zero_grad()
                #     loss.backward()
                #     optimizer.step()
                # z_list.append(zz)
                # loss_list.append(float(loss_op(zz)))
            # print(z_list)
            print("Generated 10 samples")
            print(org_loss_list)
            # print(loss_list)
            print(mini_list)

            # Generate samples using z_list, label them, add them to the labeled dataset
            for z in z_list:
                fake = self.gen(z)
                fake_flatten = torch.flatten(torch.squeeze(fake, 0), start_dim=1)
                out = self.ora(fake.detach().cpu())
                label = out.data.max(1)[1]
                # TODO: prob > threshold?
                self.labeled_data_gaal = torch.cat((self.labeled_data_gaal, fake_flatten.detach().cpu()))
                self.labeled_targets_gaal = torch.cat((self.labeled_targets_gaal, torch.tensor([label])))
            # print(labeled_data.shape)
            # print(labeled_targets.shape)
            num_label += cnt

            # re-train the SVM, update W and b
            ac, self.svm_W_tensor, self.svm_b = self.train_svm(self.labeled_data_gaal, self.labeled_targets_gaal)
            self.acc_hist_gaal.append(ac)

        # plot(self.numlabel_hist, self.acc_hist, self.args.dset, self.args.limit, self.id, self.rt)
        return self.acc_hist_gaal

    def train_random(self, unlabeled_data, unlabeled_targets):
        limit = self.args.limit
        num_label = 50
        # init linear SVM
        ac, self.svm_W_tensor, self.svm_b = self.train_svm(self.labeled_data_random, self.labeled_targets_random)
        self.acc_hist_random.append(ac)
        l_env = range(unlabeled_data.shape[0])
        while num_label < limit:
            # randomly select 10 samples from the unlabeled pool, add them to the labeled dataset
            l = random.sample(l_env, 10)
            batch_data, batch_targets = unlabeled_data[l], unlabeled_targets[l]
            l_env = [i for i in l_env if i not in l]
            self.labeled_data_random = torch.cat((self.labeled_data_random, batch_data))
            self.labeled_targets_random = torch.cat((self.labeled_targets_random, batch_targets))
            num_label += 10

            # re-train the SVM, update W and b
            ac, self.svm_W_tensor, self.svm_b = self.train_svm(self.labeled_data_random, self.labeled_targets_random)
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
    # MNIST 5 and 7 TODO: CIFAR-10 & Automobile And Horse
    dataset = dset.MNIST(root=dataroot, transform=transforms.ToTensor(), download=True)
    idx = (dataset.targets == 5) | (dataset.targets == 7)
    dataset.data, dataset.targets = torch.flatten(dataset.data[idx], start_dim=1), dataset.targets[idx]
    # print(dataset.data.shape)

    if args.dset == 'mnist57':
        test_dataset = dset.MNIST(root=dataroot, train=False, download=True, transform=transforms.ToTensor())
        idx = (test_dataset.targets == 5) | (test_dataset.targets == 7)
        test_data, test_targets = torch.flatten(test_dataset.data[idx], start_dim=1), test_dataset.targets[idx]
        # print(test_dataset.data.shape)  # [1920, 784]
    elif args.dset == 'USPS':
        test_dataset2 = dset.USPS(root=dataroot, train=True, download=True, transform=transforms.ToTensor())
        idx = [i for i in range(len(test_dataset2.targets)) if
               test_dataset2.targets[i] == 5 or test_dataset2.targets[i] == 7]
        test_dataset2.data, test_targets = np.take(test_dataset2.data, idx, 0), np.take(test_dataset2.targets, idx,
                                                                                        0)
        test_data = []
        newsize = (28, 28)
        for i in range(len(test_dataset2.data)):
            res = cv2.resize(test_dataset2.data[i].copy(), dsize=newsize, interpolation=cv2.INTER_CUBIC)
            test_data.append(res)
        test_data = np.reshape(test_data, (len(test_data), -1))

    ##################################################################

    # randomly select 50 data as the initial labeled dataset
    l = random.sample(range(dataset.data.shape[0]), 50)
    labeled_data, labeled_targets = dataset.data[l], dataset.targets[l]
    # print(labeled_data.shape) # [50, 784]
    l_env = [i for i in range(dataset.data.shape[0]) if i not in l]
    unlabeled_data, unlabeled_targets = dataset.data[l_env], dataset.targets[l_env]

    gaal_list = []
    random_list = []
    full_list = []
    total_numlabel = range(50, args.limit + 10, 10)
    # averaged over 10 runs
    id = generate_run_id()
    for run in range(10):
        print('Now run {}/{}'.format(run + 1, 10))
        oneTrain = TrainLoop(args, device, labeled_data, labeled_targets, test_data, test_targets, id, rt=run + 1)
        gaal_list.append(oneTrain.train_gaal())
        random_list.append(oneTrain.train_random(unlabeled_data, unlabeled_targets))
        # print(dataset.data.shape)
        full_acc, a, b = oneTrain.train_svm(dataset.data, dataset.targets)
        full_list.append(full_acc)

    plot_all(total_numlabel, gaal_list, random_list,full_list, args.dset, args.limit, id)
    # plot_err(total_numlabel, aver, sd, args.dset, args.limit, id)
