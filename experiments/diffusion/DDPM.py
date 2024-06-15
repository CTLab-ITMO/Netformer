import torch
from torch import nn

from experiments.diffusion.NoiserModel import NoiserModel
from experiments.diffusion.UNet import MyUNet


class DDPM:
    def __init__(self, time_steps_number, device):
        super(DDPM, self).__init__()
        self.n_steps = time_steps_number
        self.device = device
        min_beta, max_beta = 1e-4, 1e-2
        self.noiser_model = NoiserModel(min_beta, max_beta, time_steps_number, device)
        self.noise_predictor = MyUNet(time_steps_number, 100).to(device)
        self.mse = nn.MSELoss()

    def add_noise_on_image(self, x0, t, eta=None):
        return self.noiser_model.noise(x0, t, eta)

    def predict_noise(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.noise_predictor(x, t)

    def loss(self, x0, time_step):
        batch_size = len(x0)

        noise = torch.randn_like(x0).to(self.device)

        noisy_imgs = self.add_noise_on_image(x0, time_step, noise)

        noise_pred = self.predict_noise(noisy_imgs, time_step.reshape(batch_size, -1))

        return self.mse(noise_pred, noise)

    @torch.inference_mode()
    def generate_matrix(self, n_samples=25, initial_data=None):
        c, h, w = 1, 64, 64

        # Starting from random noise
        if initial_data is None:
            x = torch.randn(n_samples, c, h, w).to(self.device)
        else:
            x = initial_data

        down_steps = list(range(self.n_steps))[::-1]
        for idx, t in enumerate(down_steps):
            time_tensor = (torch.ones(n_samples) * t).to(self.device).long()
            eta_theta = self.predict_noise(x, time_tensor)
            x = self.noiser_model.denoise(x, t, eta_theta)
        return x

    def train(self):
        self.noise_predictor.train()

    def eval(self):
        self.noise_predictor.eval()
