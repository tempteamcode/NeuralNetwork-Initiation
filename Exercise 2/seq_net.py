import glob
import os
import numpy as np
from numpy import genfromtxt
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter

from dataset_seq import *

STATS_INTERVAL = 20
STATS_INTERVAL_VALID = False
BATCHSIZE=250

useGPU = True

path = "./mini_balls_seq"
valid_dataset = Balls_CF_Seq (path, 5000, 7000) 
valid_loader = torch.utils.data.DataLoader(valid_dataset,
	batch_size=BATCHSIZE, shuffle=True)

train_dataset = Balls_CF_Seq (path, 0, 5000)
train_loader = torch.utils.data.DataLoader(train_dataset,
	batch_size=BATCHSIZE, shuffle=True)

class seqNet(nn.Module):
    def __init__(self, input_dim, output_size, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
	def __init__(self):
		super(seqNet, self).__init__()
		
		input_dim = 9*4
		hidden_dim = 10
		n_layers = 1
		
		self.lstm = torch.nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
		
		hidden_state = torch.randn(n_layers, BATCHSIZE, hidden_dim)
		cell_state = torch.randn(n_layers, batch_size, hidden_dim)
		self.hidden = (hidden_state, cell_state)

	def forward(self, x):
		x, self.hidden = self.lstm(x, self.hidden)
		return torch.sigmoid(x)

model = seqNet(19*9*4, 9*4, 10, 1)
if useGPU: model = model.to("cuda:0")

# This criterion combines LogSoftMax and NLLLoss in one single class.
crossentropy = torch.nn.BCELoss()

# Set up the optimizer: stochastic gradient descent
# with a learning rate of 0.01
optimizer = torch.optim.Adam(model.parameters())

	vloss=0
	vcorrect=0
	vcount=0
	vcorrect_all=0
	vcount_all=0
	for batch_idx, (_, data) in enumerate(dataloader):
		if useGPU: data = data[:19].to("cuda:0") else data = data[:19]
		if useGPU: labels = data[19].to("cuda:0") else labels = data[19]
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

# Cycle through epochs
model.train()
for epoch in range(100):
	h = model.init_hidden(BATCHSIZE)
	# Cycle through batches
	for batch_idx, (_, data) in enumerate(train_loader):
		if useGPU: inputs = data[:19].to("cuda:0") else inputs = data[:19]
		if useGPU: labels = data[19].to("cuda:0") else labels = data[19]
	
		model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

		# Print statistics
		if (batch_idx % STATS_INTERVAL) == 0:
			print('training loss', loss)
			model.eval()
			for batch_idx, (_, data) in enumerate(valid_loader):
				if useGPU: inputs = data[:19].to("cuda:0") else inputs = data[:19]
				if useGPU: labels = data[19].to("cuda:0") else labels = data[19]
				h = tuple([each.data for each in h])
				output, h = model(inputs, h)
				valid_loss = criterion(output.squeeze(), labels.float())
				valid_losses.append(valid_loss.item())
				pred = torch.round(output.squeeze()) 
				correct_tensor = pred.eq(labels.float().view_as(pred))
				correct = np.squeeze(correct_tensor.cpu().numpy())
				num_correct += np.sum(correct)
			print('accuracy', num_correct/2000)
		
