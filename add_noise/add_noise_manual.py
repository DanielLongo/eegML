import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../data_loaders/")
sys.path.append("../forward_model/")
import stacklineplot
from scipy import signal
import random


class AddNoiseManual(object):
    def __init__(self, methods_to_use="all"):
        """
        :param methods_to_use: dict
            defines which add noise methods to include
        """
        if methods_to_use != "all":
            self.use_signal_square = methods_to_use["signal_square"]
            self.use_sin_wave = methods_to_use["sine_wave"]
        else:
            self.use_signal_square = True
            self.use_sin_wave = True

    @staticmethod
    def get_signal_square(x, b=1):
        t = np.linspace(0, x.shape[0], x.shape[1], endpoint=False)
        sig = np.sin(5 * np.pi * t)
        pwm = signal.square(t * np.pi * 5 * t)  # , duty=(sig + 1)/2)
        pwm = np.tile(pwm.reshape(768, 1), (1, x.shape[0])).T
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
        start = int(random.random() * (x.shape[1] - window_len))
        end = start + window_len
        adding_coef = [0] * start + [1] * window_len + [0] * (x.shape[1] - end)
        assert (len(adding_coef) == x.shape[1])
        adding_coef = np.asarray(adding_coef).reshape(768, 1)
        adding_coef = np.tile(adding_coef, (1, x.shape[0])).T
        adding *= adding_coef
        x += adding
        return x

    def forward(self, x):
        if self.use_signal_square:
            noise = self.get_signal_square(x, b=3)
            x = self.apply_at_random_loc(x, noise, 400)
        if self.use_sin_wave:
            noise = self.get_sine_wave(x, freq_coef=9000, b=3)
            x = self.apply_at_random_loc(x, noise, 400)
        return x

if __name__ == "__main__":
    x = np.random.randn(200, 768)
    noise_adder = AddNoiseManual