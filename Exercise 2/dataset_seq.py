import numpy as np
from skimage import io
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

COLORS = ['red', 'green', 'blue', 'yellow', 'lime', 'purple', 'orange', 'cyan', 'magenta']

class Balls_CF_Seq(Dataset):
	def __init__(self, dir, start, end, seq_count):
		self.dir = dir
		self.seq_count = end - start
		self.start = start

	# The access is _NOT_ shuffled. The Dataloader will need
	# to do this.
	def __getitem__(self, index):
		index = index + self.start
		# Load presence 
		p = np.load("%s/p_%05d.npy"%(self.dir,index))
		# Load bounding boxes and split it up
		bb = np.load("%s/seq_bb_%05d.npy"%(self.dir,index))
		return p, bb

	# Return the dataset size
	def __len__(self):
		return self.seq_count
		
if __name__ == "__main__":
	train_dataset = Balls_CF_Seq ("./mini_balls_seq", 0, 7000)

	p,b = train_dataset.__getitem__(42)

	print ("Presence:")
	print (p)

	print ("Pose:")
	print (b)