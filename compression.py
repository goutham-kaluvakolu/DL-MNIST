import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(root = "./data" , train = True , download = True ,  transform = transform)
valid_dataset = torchvision.datasets.MNIST(root = "./data" , train = False , download = True ,  transform = transform)

train_dl = torch.utils.data.DataLoader(train_dataset , batch_size = 16)
val = torch.utils.data.DataLoader(valid_dataset , batch_size = 16)

class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_dimensions, data_shape):
        super(SimpleAutoencoder, self).__init__()
        self.latent_dimensions = latent_dimensions
        self.data_shape = data_shape

        # Encoder with dropout
        self.encoder = nn.Sequential(
            nn.Linear(784, 194),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(194, 97),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(97, latent_dimensions),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dimensions, 97),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(97, 194),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(194, data_shape[0] * data_shape[1]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded_data = self.encoder(x)
        decoded_data = self.decoder(encoded_data)
        return decoded_data

latent_dimensions = 48
data_shape = (28, 28)
model = SimpleAutoencoder(latent_dimensions, data_shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

train_loss = []
valid_loss = []

num_epochs = 100

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

for epoch in range(num_epochs):
  train_epoch_loss = 0
  dev_epoch_loss = 0
  for (imgs , _) in train_dl:
    imgs = imgs.to(device)
    imgs = imgs.flatten(1)
    output = model(imgs)
    loss = loss_fn(output.view(-1,784) , imgs)
    train_epoch_loss += loss.cpu().detach().numpy()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  scheduler.step()
  print(f"epoch {epoch} loss {train_epoch_loss/len(train_dl)}")
  train_loss.append(train_epoch_loss/len(train_dl))
  with torch.no_grad():
    for (imgs , _) in val:
      imgs = imgs.to(device)
      imgs = imgs.flatten(1)
      output = model(imgs)
      loss = loss_fn(output.view(-1,784) , imgs)
      dev_epoch_loss += loss.cpu().detach().numpy()
    print(f"dev loss {dev_epoch_loss/len(val)}")
    valid_loss.append(dev_epoch_loss/len(val))
torch.save(model.state_dict(), 'model_weights.pth')

torch.save(model.state_dict(), 'model_weights.pth')


import matplotlib.pyplot as plt

plt.plot(train_loss, color='green', label='train_loss')
plt.plot(valid_loss, color='red', label='valid_loss')
plt.legend()
plt.show()


import matplotlib.pyplot as plt

# Get the first 5 images from the validation set
images = valid_dataset.data[:5]
# images = train_dataset.data[:5]

images = images.to(device)
# images = images.flatten(1)
images =  images/255.0

# Get the labels for the first 5 images
# Pass the images through the autoencoder
encoded_images = model.encoder(images.view(-1, 784))
decoded_images = model.decoder(encoded_images)

# Plot the original and decoded images side by side
for i in range(5):
  plt.subplot(2, 5, i + 1)
  plt.imshow(images[i].numpy().reshape(28, 28), cmap='gray')
  plt.axis('off')
  plt.subplot(2, 5, i + 6)
  plt.imshow(decoded_images[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
  plt.axis('off')
plt.show()

from torchsummary import summary
summary(model, (1, 28*28))

model.eval()

latent_vectors = []

with torch.no_grad():
  for imgs, _ in train_dl:
    imgs = imgs.to(device)
    imgs = imgs.flatten(1)
    latent_vectors.append(model.encoder(imgs))

# Concatenate the latent vectors into a single tensor
latent_vectors = torch.cat(latent_vectors, dim=0)

print(f"train dataset shape: {train_dataset.data.shape}, {60000 * 28 * 28}")
print(f"latent vectors shape: {latent_vectors.shape}, is {60000 * 48}")
print(f"img comressed: {28 * 28 / 48} times")