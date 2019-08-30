import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("/mnt/home2/dlongo/eegML/data_loaders/")
sys.path.append("/mnt/home2/dlongo/eegML/forward_model/")
from scipy import signal
import random
from forward_model_dataloader import ForwardModelDataset


class AddNoiseManual(object):
    def __init__(self, methods_to_use="all", b=3):
        """
        :param methods_to_use: dict
            defines which add noise methods to include
        """
        self.b = b
        if methods_to_use != "all":
            self.use_signal_square = methods_to_use["signal_square"]
            self.use_sin_wave = methods_to_use["sine_wave"]
        else:
            self.use_signal_square = True
            self.use_sin_wave = True

    @staticmethod
    def get_signal_square(x, b=1):
        n_channels = x.shape[0]
        n_samples = x.shape[1]
        t = np.linspace(0, n_channels, n_samples, endpoint=False)
        sig = np.sin(5 * np.pi * t)
        pwm = signal.square(t * np.pi * 5 * t)  # , duty=(sig + 1)/2)
        pwm = np.tile(pwm.reshape(n_samples, 1), (1, n_channels)).T
        return pwm * b

    @staticmethod
    def get_sine_wave(x, freq_coef, b=1):
        n_channels = x.shape[0]
        n_samples = x.shape[1]
        z = np.linspace(0, n_channels, n_samples, endpoint=False) * freq_coef
        noise = np.sin(z)
        noise = np.tile(noise.reshape(n_samples, 1), (1, n_channels)).T
        return noise * b

    @staticmethod
    def apply_at_random_loc(x, adding, window_len):
        n_channels = x.shape[0]
        n_samples = x.shape[1]
        start = int(random.random() * (n_samples - window_len))
        end = start + window_len
        adding_coef = [0] * start + [1] * window_len + [0] * (n_samples - end)
        assert (len(adding_coef) == x.shape[1])
        adding_coef = np.asarray(adding_coef).reshape(n_samples, 1)
        adding_coef = np.tile(adding_coef, (1, n_channels)).T
        adding *= adding_coef
        x += adding
        return x

    def __call__(self, x):
        if type(x) != np.ndarray: 
            x = x.numpy()
        if self.use_signal_square:
            noise = self.get_signal_square(x, b=self.b)
            x = self.apply_at_random_loc(x, noise, 400)
        if self.use_sin_wave:
            noise = self.get_sine_wave(x, freq_coef=9000, b=self.b)
            x = self.apply_at_random_loc(x, noise, 400)
        return x

if __name__ == "__main__":
    estimated_eegs = ForwardModelDataset(10, batch_size=2, save_source=True)
    x = estimated_eegs.getEEGs(0)[0]
    noise_adder = AddNoiseManual()
    x_noisy = noise_adder(x)
    print(x_noisy.shape)


