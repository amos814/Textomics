from torch.utils.data import Dataset
import numpy as np
import torch
class GEO(Dataset):
	def __init__(self, tokenizer, data):
		self.data = data
		self.tokenizer = tokenizer
	def __len__(self):
		return len(self.data)

	def clean_text(self, text):
		text = text.replace('\n','')
		text = text.replace('``', '')
		text = text.replace('"', '')

	def convert_to_features(self, batch):
		vec = batch[0]
		text = batch[1]
		vec = [float(x) for x in vec]
		vec = np.array(vec)
		vec = torch.from_numpy(vec)
		return text, vec

	def __getitem__(self,index):
		data, vec= self.convert_to_features(self.data[index])
		return {"data": data, "vec": vec}
class Text2Data(Dataset):
	def __init__(self, data):
		self.data = data
	def __len__(self):
		return len(self.data)

	def convert_to_features(self, batch):
		vec = batch[0]
		emb = batch[1]
		return vec, emb

	def __getitem__(self,index):
		vec, emb= self.convert_to_features(self.data[index])
		return {"vec": vec, "emb": emb}
class KNN_LM(Dataset):
	def __init__(self, tokenizer, data):
		self.data = data
		self.tokenizer = tokenizer
	def __len__(self):
		return len(self.data)

	def clean_text(self, text):
		text = text.replace('\n','')
		text = text.replace('``', '')
		text = text.replace('"', '')

	def convert_to_features(self, batch):
		target = batch[0]
		source = batch[1]
		target = target
		source = source
		return source, target

	def __getitem__(self,index):
		source, target= self.convert_to_features(self.data[index])
		return {"source": source, "target": target}
		

