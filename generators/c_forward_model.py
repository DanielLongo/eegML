import torch
from torch import nn

from conditional_generator import ConditionalGenerator

from forward_model_G import ForwardModelEnabledG

class cGForwardModel(nn.Module):
	def __init__(self, num_nodes, d, y_input_size, num_layers=3, bidirectional=True, fs_gen=200):
		super(cGForwardModel, self).__init__()
		self.y_input_size = y_input_size

		self.forward_model = ForwardModelEnabledG(num_nodes, d, num_layers=num_layers, bidirectional=bidirectional, fs_gen=fs_gen)
		self.sensor_space_net = ConditionalGenerator(num_nodes, d, y_input_size, num_layers=num_layers, bidirectional=bidirectional)

	def forward(self, x, y):
		sensor_space = self.forward_model(x)
		out = self.sensor_space_net(sensor_space, y)
		return out

	def generate_noise(self, batch_size, num_signals, num_nodes,  num_dipoles=7498):
		return [torch.randn(batch_size, num_signals, num_dipoles), torch.randn(batch_size, num_signals, self.y_input_size)]

if __name__ == "__main__":
	g = cGForwardModel(5, 50, 40)
	args = g.generate_noise(4, 100, 44)
	x = args[0]
	y = args[1]
	print(g(x, y).shape)
