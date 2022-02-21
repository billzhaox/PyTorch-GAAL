import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.datasets as dset
from utils import parse_args, plot
from gan.dcgan import Generator
from oracle.cnn import CNN
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn import svm
from oracle import cnn
import random
from sklearn.metrics import accuracy_score

# random.seed(1)

# specify the device on which the model will be trained
TRAINING_ON_GPU = torch.cuda.is_available()
device = torch.device("cuda:0" if TRAINING_ON_GPU else "cpu")
print('GPU available: ', TRAINING_ON_GPU)

################### Prepare data #################################
args = parse_args()
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
    test_dataset2 = dset.USPS(root=dataroot, train=False, download=True, transform=transforms.ToTensor())
    idx = [i for i in range(len(test_dataset2.targets)) if test_dataset2.targets[i] == 5 or test_dataset2.targets[i] == 7]
    test_dataset2.data, test_targets = np.take(test_dataset2.data, idx, 0), np.take(test_dataset2.targets, idx, 0)
    test_data = []
    newsize = (28, 28)
    for i in range(len(test_dataset2.data)):
        res = cv2.resize(test_dataset2.data[i].copy(), dsize=newsize, interpolation=cv2.INTER_CUBIC)
        test_data.append(res)
    test_data = np.reshape(test_data, (len(test_data), -1))

# randomly select 50, and throw away other labels
l = random.sample(range(dataset.data.shape[0]), 50)
l_env = [i for i in range(dataset.data.shape[0]) if i not in l]
labeled_data = dataset.data[l]
# print(labeled_data.shape) # [50, 784]
labeled_targets = dataset.targets[l]
# print(labeled_targets.shape) # [50]
# dataset.data, dataset.targets = dataset.data[l_env], dataset.targets[l_env]
unlabeled_set = dataset.data[l_env]
num_label = 50
# init SVM
clf = svm.SVC(kernel='linear', gamma=0.001)
acc_hist = []
numlabel_hist = []

def train_svm():
    clf.fit(labeled_data, labeled_targets)
    W = clf.coef_[0]  # W consists of 28*28=784 elements
    b = clf.intercept_[0]  # b consists of 1 element
    W_tensor = torch.from_numpy(W).float().to(device)
    W_tensor.requires_grad_(True)
    pred_labels = clf.predict(test_data)
    acc = accuracy_score(test_targets, pred_labels)
    # print(acc)
    acc_hist.append(acc)
    numlabel_hist.append(num_label)
    # plt.plot(numlabel_hist,acc_hist,'o-')
    # plt.show()
    return W_tensor, b


svm_W_tensor, svm_b = train_svm()


# print(pred_labels.shape)

# # print('data_size: {}, batch_size: {}, latent_size: {}'.format(data_size, batch_size, latent_size))

# Load generator G
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
gen = Generator(ngpu).to(device)
gen.load_state_dict(torch.load('./gan/generator/G.pth'))
gen.eval()


# def show(imgs):
#     if not isinstance(imgs, list):
#         imgs = [imgs]
#     fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
#     for i, img in enumerate(imgs):
#         img = img.detach()
#         img = F.to_pil_image(img).show()
#         plt.imshow(img)
#         axs[0, i].imshow(np.asarray(img))
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])



# print(input)
# grid = vutils.make_grid(fake, padding=2, normalize=True)
# show(grid)

# loss function we want to optimize
def loss_op(z):
    fake = gen(z)
    # print(fake.shape)  # 1*1*28*28
    fake_res = torch.flatten(torch.squeeze(fake))  # G(z)
    loss = torch.dot(svm_W_tensor, fake_res) + svm_b  # W * G(z) + b
    return torch.abs(loss)


# Load oracle classifer
ora = CNN()
device_type = "GPU" if torch.cuda.is_available() else "CPU"
ora.load_state_dict(torch.load("./oracle/CNN_mnist57.pth".format(device_type)))

# for i in range(len(labeled_data)):
#     dd = torch.reshape(labeled_data[i],(1,1,28,28))
#     dd = dd / 255.
#     # print(dd.shape)
#     # print(dd)
#     out = ora(dd)
#
#     label = out.data.max(1)[1]
#     print("  {}            {}".format(int(label), labeled_targets[i]))


# training loop
limit = args.limit
while num_label < limit:

    print('Now label {}/{}'.format(num_label, limit))

    # Solve optimization problem by GD, 10 queries every time
    z_list = []
    org_loss_list = []
    loss_list = []
    iterr = 500
    learning_rate_init = 1000.
    alpha = 0.5
    for q in range(10):
        history = []
        history_lr = []
        zz = torch.randn(1, 100, 1, 1, device=device, requires_grad=True)
        org_loss_list.append(float(loss_op(zz)))
        optimizer = torch.optim.SGD([zz], lr=1., momentum=0.9)
        for i in range(iterr):
            loss = loss_op(zz)
            history.append(loss.detach().cpu().clone().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        z_list.append(zz)
        loss_list.append(float(loss_op(zz)))
    # print(z_list)
    print("Generated 10 samples")
    print(org_loss_list)
    print(loss_list)

    # Generate samples using z_list, label them, add them to the labeled dataset
    for z in z_list:
        fake = gen(z)
        fake_flatten = torch.flatten(torch.squeeze(fake, 0), start_dim=1)
        out = ora(fake.detach().cpu())
        label = out.data.max(1)[1]
        # TODO: prob > threshold?
        labeled_data = torch.cat((labeled_data, fake_flatten.detach().cpu()))
        labeled_targets = torch.cat((labeled_targets, torch.tensor([label])))
    # print(labeled_data.shape)
    # print(labeled_targets.shape)
    num_label += 10

    # re-train the SVM, update W and b
    svm_W_tensor, svm_b = train_svm()

plot(numlabel_hist, acc_hist, args.dset)


