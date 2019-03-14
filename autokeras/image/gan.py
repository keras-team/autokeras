import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from copy import deepcopy
from torchvision import utils as vutils
from tqdm import tqdm

from autokeras.backend import Backend
from autokeras.constant import Constant
from autokeras.nn.model_trainer import ModelTrainerBase
from autokeras.unsupervised import Unsupervised


class DCGAN(Unsupervised):
    """ Deep Convolution Generative Adversary Network
    """

    def __init__(self, nz=100, ngf=32, ndf=32, nc=3, verbose=False, gen_training_result=None,
                 augment=None):
        """
       Args:
            nz: size of the latent z vector
            ngf: of gen filters in first conv layer
            ndf: of discrim filters in first conv layer
            nc: number of input chanel
            verbose: A boolean of whether the search process will be printed to stdout.
            gen_training_result: A tuple of (path, size) to denote where to output the intermediate result with size
            augment: A boolean value indicating whether the data needs augmentation.
        """
        super().__init__(verbose)
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc
        self.verbose = verbose
        self.device = Backend.get_device()
        self.gen_training_result = gen_training_result
        self.augment = augment if augment is not None else Constant.DATA_AUGMENTATION
        self.data_transformer = None
        self.net_d = Discriminator(self.nc, self.ndf)
        self.net_g = Generator(self.nc, self.nz, self.ngf)

    def fit(self, x_train):
        """ Train only

        Args:
            x_train: ndarray contained the training data

        Returns:

        """
        # input size stay the same, enable  cudnn optimization
        cudnn.benchmark = True
        self.data_transformer = Backend.get_image_transformer(x_train, augment=self.augment)
        train_dataloader = self.data_transformer.transform_train(x_train)
        GANModelTrainer(self.net_g,
                        self.net_d,
                        train_dataloader,
                        Backend.binary_classification_loss,
                        self.verbose,
                        self.gen_training_result,
                        device=Backend.get_device()).train_model()

    def generate(self, input_sample=None):
        if input_sample is None:
            input_sample = torch.randn(self.gen_training_result[1], self.nz, 1, 1, device=self.device)
        if not isinstance(input_sample, torch.Tensor) and \
                isinstance(input_sample, np.ndarray):
            input_sample = torch.from_numpy(input_sample)
        if not isinstance(input_sample, torch.Tensor) and \
                not isinstance(input_sample, np.ndarray):
            raise TypeError("Input should be a torch.tensor or a numpy.ndarray")
        self.net_g.eval()
        with torch.no_grad():
            input_sample = input_sample.to(self.device)
            generated_fake = self.net_g(input_sample)
        vutils.save_image(generated_fake.detach(),
                          '%s/evaluation.png' % self.gen_training_result[0],
                          normalize=True)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        output = self.main(input_tensor)
        return output.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input_tensor):
        output = self.main(input_tensor)
        return output


class GANModelTrainer(ModelTrainerBase):
    """A ModelTrainer especially for the GAN.
    Attributes:
        d_model: A discriminator model.
        g_model: A generator model.
        out_f: Out file.
        out_size: Size of the output image.
        optimizer_d: Optimizer for discriminator.
        optimizer_g: Optimizer for generator.
    """

    def __init__(self,
                 g_model,
                 d_model,
                 train_data,
                 loss_function,
                 verbose,
                 gen_training_result=None,
                 device=None):
        """Initialize the GANModelTrainer.
        Args:
            g_model: The generator model to be trained.
            d_model: The discriminator model to be trained.
            train_data: the training data.
            loss_function: The loss function for both discriminator and generator.
            verbose: Whether to output the system output.
            gen_training_result: Whether to generate the intermediate result while training.
        """
        super().__init__(loss_function, train_data, verbose=verbose, device=device)
        self.d_model = d_model
        self.g_model = g_model
        self.d_model.to(self.device)
        self.g_model.to(self.device)
        self.out_f = None
        self.out_size = 0
        if gen_training_result is not None:
            self.out_f, self.out_size = gen_training_result
            self.sample_noise = torch.randn(self.out_size,
                                            self.g_model.nz,
                                            1, 1, device=self.device)
        self.optimizer_d = None
        self.optimizer_g = None

    def train_model(self,
                    max_iter_num=None,
                    max_no_improvement_num=None,
                    timeout=None):
        if max_iter_num is None:
            max_iter_num = Constant.MAX_ITER_NUM
        self.optimizer_d = torch.optim.Adam(self.d_model.parameters())
        self.optimizer_g = torch.optim.Adam(self.g_model.parameters())
        if self.verbose:
            progress_bar = tqdm(total=max_iter_num,
                                desc='     Model     ',
                                file=sys.stdout,
                                ncols=75,
                                position=1,
                                unit=' epoch')
        else:
            progress_bar = None
        for epoch in range(max_iter_num):
            self._train(epoch)
            if self.verbose:
                progress_bar.update(1)
        if self.verbose:
            progress_bar.close()

    def _train(self, epoch):
        """Perform the actual train."""
        # put model into train mode
        self.d_model.train()
        # TODO: why?
        cp_loader = deepcopy(self.train_loader)
        if self.verbose:
            progress_bar = tqdm(total=len(cp_loader),
                                desc='Current Epoch',
                                file=sys.stdout,
                                leave=False,
                                ncols=75,
                                position=0,
                                unit=' Batch')
        else:
            progress_bar = None
        real_label = 1
        fake_label = 0
        for batch_idx, inputs in enumerate(cp_loader):
            # Update Discriminator network maximize log(D(x)) + log(1 - D(G(z)))
            # train with real
            self.optimizer_d.zero_grad()
            inputs = inputs.to(self.device)
            batch_size = inputs.size(0)
            outputs = self.d_model(inputs)

            label = torch.full((batch_size,), real_label, device=self.device)
            loss_d_real = self.loss_function(outputs, label)
            loss_d_real.backward()

            # train with fake
            noise = torch.randn((batch_size, self.g_model.nz, 1, 1,), device=self.device)
            fake_outputs = self.g_model(noise)
            label.fill_(fake_label)
            outputs = self.d_model(fake_outputs.detach())
            loss_g_fake = self.loss_function(outputs, label)
            loss_g_fake.backward()
            self.optimizer_d.step()
            # (2) Update G network: maximize log(D(G(z)))
            self.g_model.zero_grad()
            label.fill_(real_label)
            outputs = self.d_model(fake_outputs)
            loss_g = self.loss_function(outputs, label)
            loss_g.backward()
            self.optimizer_g.step()

            if self.verbose:
                if batch_idx % 10 == 0:
                    progress_bar.update(10)
            if self.out_f is not None and batch_idx % 100 == 0:
                fake = self.g_model(self.sample_noise)
                vutils.save_image(
                    fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (self.out_f, epoch),
                    normalize=True)
        if self.verbose:
            progress_bar.close()