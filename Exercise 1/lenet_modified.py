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
from numpy import genfromtxt
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#from torch.utils.tensorboard import SummaryWriter

from dataset_det import *

STATS_INTERVAL = 5
BATCHSIZE=250

useGPU = True

path = "./mini_balls/train"
valid_dataset = Balls_CF_Detection (path, 16000, 21000) 
valid_loader = torch.utils.data.DataLoader(valid_dataset,
	batch_size=BATCHSIZE, shuffle=True)

train_dataset = Balls_CF_Detection (path, 0, 16000)
train_loader = torch.utils.data.DataLoader(train_dataset,
	batch_size=BATCHSIZE, shuffle=True)



class LeNet(torch.nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = torch.nn.Conv2d(3, 3, 5, 1)
		self.conv2 = torch.nn.Conv2d(3, 5, 3, 1)
		self.fc1 = torch.nn.Linear(23*23*5, 50)
		self.fc2 = torch.nn.Linear(50, 9)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 23*23*5)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return torch.sigmoid(x)

model = LeNet()
if useGPU: model = model.to("cuda:0")

# This criterion combines LogSoftMax and NLLLoss in one single class.
crossentropy = torch.nn.BCELoss()

# Set up the optimizer: stochastic gradient descent
# with a learning rate of 0.01
optimizer = torch.optim.Adam(model.parameters())

# Setting up tensorboard
# writer = SummaryWriter('./ball_detec')

# ************************************************************************
# Calculate the error of a model on data from a given loader
# This is used to calculate the validation error every couple of
# thousand batches
# ************************************************************************

def calcError (net, dataloader):
	vloss=0
	vcorrect=0
	vcount=0
	vcorrect_all=0
	vcount_all=0
	for batch_idx, (data, labels, _) in enumerate(dataloader):
		if useGPU: data = data.to("cuda:0")
		if useGPU: labels = labels.to("cuda:0")
		y = model(data)
		loss = crossentropy(y, labels)
		vloss += loss.item()
		predicted = torch.round(y.data)
		#print(predicted.size())
		#print(labels.size())
		vcorrect += (predicted == labels).sum().item()
		vcount += BATCHSIZE*9
		vcorrect_all += sum(row.sum().item() == 9 for row in (predicted == labels))
		vcount_all += BATCHSIZE
	return vloss/len(dataloader), 100.0*(1.0-vcorrect/vcount), 100.0*(1.0-vcorrect_all/vcount_all)

# Training
running_loss = 0.0
running_correct = 0
running_count = 0
running_correct_all = 0
running_count_all = 0

# Add the graph to tensorboard
dataiter = iter(train_loader)
data, labels, _ = dataiter.next()
# writer.add_graph (model, data)
# writer.flush()

# Cycle through epochs
for epoch in range(100):
	
	# Cycle through batches
	for batch_idx, (data, labels, _) in enumerate(train_loader):
		if useGPU: data = data.to("cuda:0")
		if useGPU: labels = labels.to("cuda:0")
	
		optimizer.zero_grad()
		y = model(data)
		loss = crossentropy(y, labels)
		loss.backward()
		running_loss += loss.item()
		optimizer.step()

		predicted = torch.round(y.data) #arrondi vers valeur paire (0.5 -> 0)
		running_correct += (predicted == labels).sum().item()
		running_count += BATCHSIZE*9
		running_correct_all += sum(row.sum().item() == 9 for row in (predicted == labels))
		running_count_all += BATCHSIZE

		# Print statistics
		if (batch_idx % STATS_INTERVAL) == 0:
			train_err = 100.0*(1.0-running_correct / running_count)
			train_err_all = 100.0*(1.0-running_correct_all / running_count_all)
			valid_loss, valid_err, valid_err_all = calcError(model, valid_loader)
			print ('Epoch: %d batch: %5d' % (epoch + 1, batch_idx + 1), end="")
			print (' train-loss: %.3f train-err: %.3f %3.f' % (running_loss / STATS_INTERVAL, train_err, train_err_all), end="")
			print (' valid-loss: %.3f valid-err: %.3f %3.f' % (valid_loss, valid_err, valid_err_all))

			# Write statistics to the log file
			# writer.add_scalars ('Loss', {
				# 'training:': running_loss / STATS_INTERVAL,
				# 'validation:': valid_loss }, 
				# epoch * len(train_loader) + batch_idx)

			# writer.add_scalars ('Error', {
				# 'training:': train_err,
				# 'validation:': valid_err }, 
				# epoch * len(train_loader) + batch_idx)
							
			running_loss = 0.0
			running_correct = 0.0
			running_count=0.0

