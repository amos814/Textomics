import torch.nn as nn
import torch.nn.functional as F
import torch

class FC(nn.Module):
	def __init__(self, emb_dim, vec_dim):
		super(FC, self).__init__()
		self.fc1 = nn.Linear(emb_dim, emb_dim * 2)
		self.fc2 = nn.Linear(emb_dim * 2, vec_dim)

	def forward(self, data):
		x = F.relu(self.fc1(data))
		x = self.fc2(x)
		return x
class Embed(nn.Module):
	def __init__(self, vec_dim, emb_dim):
		super(Embed, self).__init__()
		self.fc1 = nn.Linear(vec_dim, emb_dim)
		# self.fc2 = nn.Linear(emb_dim , emb_dim)
	def forward(self, data):
		x = self.fc1(data)
		# x = self.fc2(x)
		return x