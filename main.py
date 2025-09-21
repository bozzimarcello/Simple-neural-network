import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class Model(nn.Module):
  def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x
  
torch.manual_seed(41)

model = Model()

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

my_df['species'] = my_df['species'].replace('setosa', 0.0)
my_df['species'] = my_df['species'].replace('versicolor', 1.0)
my_df['species'] = my_df['species'].replace('virginica', 2.0)

# Split the dataset in input and output
X = my_df.drop('species', axis=1)
y = my_df['species']

# Convert to numpy arrays
X = X.values
y = y.values

# Test train split for pytorch: 80 train, 20 test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert X features to float tensors (?) integers
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to long tensors (?) 64 bit integers
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set the criterionthat says how far the predictions are from the result
criterion = nn.CrossEntropyLoss()

# Choose Adam optimizer, lr = learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train our model
# Epochs: one run thru all the training data in our network
epochs = 100
losses = []
for i in range(epochs):
  # Get predictions
  y_pred = model.forward(X_train)

  # Measure the loss/error
  loss = criterion(y_pred, y_train)

  # Keep track of losses
  losses.append(loss.detach().numpy())

  # print every 10 epoch
  if i % 10 == 0:
    print(f'Epoch: {i} and loss: {loss}')

  # Do some back propagation
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()