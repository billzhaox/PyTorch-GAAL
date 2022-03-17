from __future__ import print_function
# %matplotlib inline
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from gan.dcgan import Generator, Discriminator

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

############################ Inputs ####################################
# Root directory for dataset
dataroot = '../data/'
# Number of workers for dataloader
workers = 0
# Batch size during training
batch_size = 256
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 28  # 64
# Number of channels in the training images. For color images this is 3
nc = 1  # 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 28
# Number of training epochs
num_epochs = 30  # 10
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

src = 'MNIST'
####################### Data #################################
# We can use an image folder dataset the way we have it setup.
# Create the dataset

dataset = dset.MNIST(root=dataroot, transform=transforms.ToTensor(), download=True)
idx = (dataset.targets == 5) | (dataset.targets == 7)
dataset.data, dataset.targets = dataset.data[idx], dataset.targets[idx]
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)  # range:[0,1]

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# Plot some training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

############################# Weight Initialization ##################################
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


############################# Training ##################################
x_data = np.arange(0.001, 1, 0.001)
sat_loss_data = np.log(1 - x_data)
sat_loss_derivation_data = -1 / (1 - x_data)
non_sat_loss_data = -np.log(x_data)
non_sat_loss_derivation_data = -1 / x_data

# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(x_data, sat_loss_data, 'r', label='Saturating G loss')
# ax.plot(x_data, sat_loss_derivation_data, 'r--', label='Derivation saturating G loss')
# ax.plot(x_data, non_sat_loss_data, 'b', label='non-saturating loss')
# ax.plot(x_data, non_sat_loss_derivation_data, 'b--', label='Derivation non-saturating G loss')
# ax.set_xlim([0, 1])
# ax.set_ylim([-10, 4])
# ax.grid(True, which='both')
# ax.axhline(y=0, color='k')
# ax.axvline(x=0, color='k')
# ax.set_title('Saturating and non-saturating loss function')
# plt.xlabel('D(G(z))')
# plt.ylabel('Loss / derivation of loss')
# ax.legend()
# plt.show()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)


def training_loop(num_epochs=num_epochs, saturating=False):
    ## Create the generator
    netG = Generator(ngpu, mode=0).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    ## Create the Discriminator
    netD = Discriminator(ngpu, mode=0).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    ## Initialize BCELoss function
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()  # more stable than nn.BCELoss

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    ## Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    G_grads_mean = []
    G_grads_std = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device, dtype=torch.float32)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()

            if saturating:
                label.fill_(fake_label)  # Saturating loss: Use fake_label y = 0 to get J(G) = log(1−D(G(z)))
            else:
                label.fill_(real_label)  # Non-saturating loss: fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output

            if saturating:
                errG = -criterion(output, label)  # Saturating loss: -J(D) = J(G)
            else:
                errG = criterion(output, label)  # Non-saturating loss

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Save gradients
            G_grad = [p.grad.view(-1).cpu().numpy() for p in list(netG.parameters())]
            G_grads_mean.append(np.concatenate(G_grad).mean())
            G_grads_std.append(np.concatenate(G_grad).std())

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch + 1, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    torch.save(netG.state_dict(), './generator/G_' + src + '.pth')
    # torch.save(netD.state_dict(), './discriminator/D'+src+'.pth')
    return G_losses, D_losses, G_grads_mean, G_grads_std, img_list


# G_losses_sat, D_losses_sat, G_grads_mean_sat, G_grads_std_sat, img_list_sat = training_loop(saturating=True)

G_losses_nonsat, D_losses_nonsat, G_grads_mean_nonsat, G_grads_std_nonsat, img_list_nonsat = training_loop(
    saturating=False)

############################# Results ##################################
# plt.figure(figsize=(10,5))
# plt.title("Generator and discriminator loss")
# # plt.plot(G_losses_sat,label="Saturating G loss", alpha=0.75)
# # plt.plot(D_losses_sat,label="Saturating D loss", alpha=0.75)
# plt.plot(G_losses_nonsat,label="Non-saturating G loss", alpha=0.75)
# plt.plot(D_losses_nonsat,label="Non-saturating D loss", alpha=0.75)
# plt.xlabel("Iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(10,5))
# plt.title("Generator gradient means")
# plt.plot(G_grads_mean_sat, label="Saturating G loss", alpha=0.75)
# plt.plot(G_grads_mean_nonsat, label="Non-saturating G loss", alpha=0.75)
# plt.xlabel("Iterations")
# plt.ylabel("Gradient mean")
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(10,5))
# plt.title("Generator gradient standard deviations")
# plt.plot(G_grads_std_sat,label="Saturating G loss", alpha=0.75)
# plt.plot(G_grads_std_nonsat,label="Non-saturating G loss", alpha=0.75)
# plt.xlabel("Iterations")
# plt.ylabel("Gradient standard deviation")
# plt.legend()
# plt.show()

# Grab a batch of real images from the dataloader
# real_batch = next(iter(dataloader))

# Plot the real images
# plt.figure(figsize=(15,15))
# plt.subplot(1,3,1)
# plt.axis("off")
# plt.title("Real images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
# plt.subplot(1,3,2)
# plt.axis("off")
# plt.title("Fake images - saturating G loss")
# plt.imshow(np.transpose(img_list_sat[-1],(1,2,0)))

# Plot the fake images from the last epoch
print(img_list_nonsat[-1].shape)
plt.subplot(1, 3, 3)
plt.axis("off")
plt.title("Fake images - non-saturating G loss")
plt.imshow(np.transpose(img_list_nonsat[-1], (1, 2, 0)))
plt.show()
# print(img_list_nonsat[-1])
