import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.utils as vutils

from autokeras.constant import Constant
from autokeras.loss_function import binary_classification_loss
from autokeras.model_trainer import GANModelTrainer
from autokeras.preprocessor import DataTransformer
from autokeras.unsupervised import Unsupervised
from autokeras.utils import get_device


class DCGAN(Unsupervised):
    """ Deep Convolution Generative Adversary Network
    """

    def __init__(self, nz=100, ngf=32, ndf=32, nc=3, verbose=False, gen_training_result=None,
                 augment=Constant.DATA_AUGMENTATION):
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
        self.gen_training_result = gen_training_result
        self.augment = augment
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
        self.data_transformer = DataTransformer(x_train, augment=self.augment)
        train_dataloader = self.data_transformer.transform_train(x_train)
        GANModelTrainer(self.net_g,
                        self.net_d,
                        train_dataloader,
                        binary_classification_loss,
                        self.verbose,
                        self.gen_training_result).train_model()

    def generate(self, input_sample=None):
        if input_sample is None:
            input_sample = torch.randn(self.gen_training_result[1], self.nz, 1, 1, device=get_device())
        if not isinstance(input_sample, torch.Tensor) and \
                isinstance(input_sample, np.ndarray):
            input_sample = torch.from_numpy(input_sample)
        if not isinstance(input_sample, torch.Tensor) and \
                not isinstance(input_sample, np.ndarray):
            raise TypeError("Input should be a torch.tensor or a numpy.ndarray")
        self.net_g.eval()
        with torch.no_grad():
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

    def forward(self, input):
        output = self.main(input)
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

    def forward(self, input):
        output = self.main(input)
        return output
