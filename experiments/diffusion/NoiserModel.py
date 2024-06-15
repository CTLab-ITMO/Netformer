import torch


class NoiserModel:
    def __init__(self, min_beta, max_beta, time_steps_number, device):
        self.betas = torch.linspace(min_beta, max_beta, time_steps_number).to(device)
        self.alphas = (1 - self.betas).to(device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        self.alpha_bars_sqrt = self.alpha_bars.sqrt().to(device)
        self.one_minus_alpha_bars_sqrt = (1 - self.alpha_bars).sqrt().to(device)

        self.device = device

    def noise(self, x0, t, normal_noise=None):
        if normal_noise is None:
            normal_noise = torch.randn_like(x0).to(self.device)

        noisy = self.alpha_bars_sqrt[t].reshape(-1, 1, 1, 1) * x0 + \
                self.one_minus_alpha_bars_sqrt[t].reshape(-1, 1, 1, 1) * normal_noise
        return noisy

    def true_denoise(self, img, t, normal_noise):
        shift = self.one_minus_alpha_bars_sqrt[t] * normal_noise
        x = (img - shift) / self.alpha_bars_sqrt[t]
        return x

    def denoise(self, img, t, normal_noise):
        shift = (1 - self.alphas[t]) / self.one_minus_alpha_bars_sqrt[t] * normal_noise
        x = (img - shift) / self.alphas[t].sqrt()

        if t > 0:
            z = torch.randn_like(img).to(self.device)
            sigma_t = self.betas[t].sqrt()
            x = x + sigma_t * z
        return x

if __name__ == "__main__":
    min_beta, max_beta = 1e-4, 1e-2
    time_steps_number = 50

    noiser_model = NoiserModel(min_beta, max_beta, time_steps_number, 'cpu')
    print(noiser_model)