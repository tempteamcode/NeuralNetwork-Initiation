import numpy as np
#from skimage import io
from PIL import Image#
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

COLORS = ['red', 'green', 'blue', 'yellow', 'lime', 'purple', 'orange', 'cyan', 'magenta']

class Balls_CF_Detection(Dataset):
	def __init__(self, dir, image_start, image_end=None, transform=None):
		self.transform = transform
		self.dir = dir
		if not image_end:
			image_end = image_start
			image_start = 0
		self.image_count = image_end - image_start
		self.image_start = image_start

	# The access is _NOT_ shuffled. The Dataloader will need
	# to do this.
	def __getitem__(self, index):
		index_file = index + self.image_start
		#img = io.imread("%s/img_%05d.jpg"%(self.dir,index_file))
		img = Image.open("%s/img_%05d.jpg"%(self.dir,index_file))
		img = np.asarray(img)
		img = img.astype(np.float32)
		
		# Dims in: x, y, color
		# should be: color, x, y
		img = np.transpose(img, (2,0,1))
		
		img = torch.tensor(img)
		if self.transform is not None:
			img = self.transform(img)

		# Load presence and bounding boxes and split it up
		p_bb = np.load("%s/p_bb_%05d.npy"%(self.dir,index_file))
		p  = p_bb[:,0]
		bb = p_bb[:,1:5]
		return img, p, bb

	# Return the dataset size
	def __len__(self):
		return self.image_count

def main():
	# train_dataset = Balls_CF_Detection ("./mini_balls/train", 20999,
	#	 transforms.Normalize([128, 128, 128], [50, 50, 50]))
	train_dataset = Balls_CF_Detection ("./mini_balls/train", 20999)

	img,p,b = train_dataset.__getitem__(42)

	print ("Presence:")
	print (p)

	print ("Pose:")
	print (b)

if __name__ == "__main__":
	main()
