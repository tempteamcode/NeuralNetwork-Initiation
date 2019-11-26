#!/usr/bin/python
# ***************************************************************************
# Author: Christian Wolf
# christian.wolf@insa-lyon.fr
#
# Begin: 22.9.2019
# ***************************************************************************

import glob
import os
import numpy as np
#from skimage import io
from numpy import genfromtxt
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#from torch.utils.tensorboard import SummaryWriter

from dataset_det import Balls_CF_Detection, COLORS#

STATS_INTERVAL = 300

'''
class MNISTDataset(Dataset):
	def __init__(self, dir, transform=None):
		self.no_images=0
		self.transform = transform

		arrarr = [None]*10
		for i in range(10):
			print (i)
			regex="%s/%i/*.png"%(dir,i)
			entries=glob.glob(regex)
			arr=[None]*len(entries)
			for j,filename in enumerate(entries):
				# arr[j] = torch.tensor(io.imread(filename))
				arr[j] = io.imread(filename)
				if self.transform:
					arr[j] = self.transform(arr[j])
			arrarr[i] = arr
			self.no_images = self.no_images + len(entries)

		# Flatten into a single array
		self.images = [None]*self.no_images
		self.labels = [None]*self.no_images
		g_index=0
		for i in range(10):
			for t in arrarr[i]:
				self.images[g_index] = t
				self.labels[g_index] = i
				g_index += 1

	# The access is _NOT_ shuffled. The Dataloader will need
	# to do this.
	def __getitem__(self, index):
		return self.images[index], self.labels[index]

	# Return the dataset size
	def __len__(self):
		return self.no_images
		
BATCHSIZE=50

valid_dataset = MNISTDataset ("MNIST-png/testing", 
	transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))])) # mean, std of dataset
valid_loader = torch.utils.data.DataLoader(valid_dataset,
	batch_size=BATCHSIZE, shuffle=True)

train_dataset = MNISTDataset ("MNIST-png/training", 
	transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))])) # mean, std of dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
	batch_size=BATCHSIZE, shuffle=True)
'''

BATCHSIZE=10

train_dataset = Balls_CF_Detection("./mini_balls/train", 0, 20000)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)

valid_dataset = Balls_CF_Detection("./mini_balls/train", 20000, 21000)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=True)


class LeNet(torch.nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = torch.nn.Conv2d(3, 12, 5, 1) #(1, 20, 5, 1)
		self.conv2 = torch.nn.Conv2d(12, 36, 5, 1) #(20, 50, 5, 1)
		self.fc1 = torch.nn.Linear(174240//10, 400) #(4*4*50, 500)
		self.fc2 = torch.nn.Linear(400, 9) #(500, 10)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 174240//10) #(-1, 4*4*50)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.sigmoid(x) #x

model = LeNet()
#model = model.to("cuda:0")

# This criterion combines LogSoftMax and NLLLoss in one single class.
#crossentropy = torch.nn.CrossEntropyLoss(reduction='mean')
crossentropy = torch.nn.BCELoss(reduction='mean')#

# Set up the optimizer: stochastic gradient descent
# with a learning rate of 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Setting up tensorboard
#writer = SummaryWriter('runs/mnist_lenet')

# ************************************************************************
# Calculate the error of a model on data from a given loader
# This is used to calculate the validation error every couple of
# thousand batches
# ************************************************************************

def calcError (net, dataloader):
	vloss=0
	vcorrect=0
	vcount=0
	for batch_idx, (data, labels, _) in enumerate(dataloader):
		y = model(data)

		tvals, tidx = torch.topk(y, 3)
		res = torch.zeros(BATCHSIZE, 9)
		res = res.scatter(1, tidx, tvals)

		loss = crossentropy(y, labels)
		vloss += loss.item()
		#_, predicted = torch.max(y.data, 1)
		for i in range(BATCHSIZE):
			vcorrect += ((((res[i] != 0) == labels[i]).sum()).item() == 9)
		#vcorrect += (predicted == labels).sum().item()
		vcount += BATCHSIZE
	return vloss/len(dataloader), 100.0*(1.0-vcorrect/vcount)


def main():
	# Training
	running_loss = 0.0
	running_correct = 0
	running_count = 0

	'''
	# Add the graph to tensorboard
	dataiter = iter(train_loader)
	data, labels, _ = dataiter.next()
	writer.add_graph (model, data)
	writer.flush()
	'''

	# Cycle through epochs
	for epoch in range(100):

		# Cycle through batches
		for batch_idx, (data, labels, _) in enumerate(train_loader):
			#data = data.to("cuda:0") 
			optimizer.zero_grad()
			y = model(data)

			tvals, tidx = torch.topk(y, 3)
			res = torch.zeros(BATCHSIZE, 9)
			res = res.scatter(1, tidx, tvals)

			loss = crossentropy(res, labels)
			loss.backward()
			running_loss += loss.cpu().item()
			optimizer.step()

			#_, predicted = torch.max(y.data.cpu(), 1)
			for i in range(BATCHSIZE):
				running_correct += ((((res[i] != 0) == labels[i]).sum()).item() == 9)
			#running_correct += (predicted == (labels == 1.)).sum().item()
			running_count += BATCHSIZE * 9

			# Print statistics
			if ((batch_idx+1) % STATS_INTERVAL) == 0:
				train_err = 100.0*(1.0-running_correct / running_count)
				valid_loss, valid_err = calcError (model, valid_loader)
				print ('Epoch: %d batch: %5d ' % (epoch + 1, batch_idx + 1), end=" ")
				print ('train-loss: %.3f train-err: %.3f' % (running_loss / STATS_INTERVAL, train_err), end=" ")
				print (' valid-loss: %.3f valid-err: %.3f' % (valid_loss, valid_err))

				'''
				# Write statistics to the log file
				writer.add_scalars ('Loss', {
					'training:': running_loss / STATS_INTERVAL,
					'validation:': valid_loss }, 
					epoch * len(train_loader) + batch_idx)

				writer.add_scalars ('Error', {
					'training:': train_err,
					'validation:': valid_err }, 
					epoch * len(train_loader) + batch_idx)
				'''

				running_loss = 0.0
				running_correct = 0.0
				running_count=0.0

if __name__ == "__main__":
	main()