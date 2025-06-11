#By Engr-M-Tahir

#library imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import lightning as L
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# Hyperparameters
latent_dim = 100
num_classes = 10
batch_size = 64
lr = 0.0002
epochs = 50

# MNIST dataset with labels
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_loader = DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)


#Generator Class
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Embed labels and concatenate with noise vector
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        return self.fc(x).view(-1, 1, 28, 28)

#Discriminator class
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Embed labels and concatenate with flattened image
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(28*28 + num_classes, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        c = self.label_emb(labels)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, c], dim=1)
        return self.fc(x)


#GAN Class

class GAN(L.LightningModule):
    def __init__(self):
        super(GAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.criterion = nn.BCELoss()
        self.automatic_optimization = False

    def forward(self, z, labels):
        return self.generator(z, labels)

    def configure_optimizers(self):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [optimizer_g, optimizer_d]

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = imgs.view(imgs.size(0), -1)

        real_labels = torch.full((imgs.size(0),), 1, dtype=torch.float, device=self.device)
        fake_labels = torch.full((imgs.size(0),), 0, dtype=torch.float, device=self.device)

        opt_g, opt_d = self.optimizers()

        # --- Train Discriminator ---
        outputs_real = self.discriminator(imgs, labels)
        d_loss_real = self.criterion(outputs_real.squeeze(), real_labels)

        z = torch.randn(imgs.size(0), latent_dim, device=self.device)
        sampled_labels = torch.randint(0, num_classes, (imgs.size(0),), device=self.device)
        fake_imgs = self.generator(z, sampled_labels)
        outputs_fake = self.discriminator(fake_imgs.detach(), sampled_labels)
        d_loss_fake = self.criterion(outputs_fake.squeeze(), fake_labels)

        d_loss = d_loss_real + d_loss_fake
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # --- Train Generator ---
        z = torch.randn(imgs.size(0), latent_dim, device=self.device)
        sampled_labels = torch.randint(0, num_classes, (imgs.size(0),), device=self.device)
        fake_imgs = self.generator(z, sampled_labels)
        outputs = self.discriminator(fake_imgs, sampled_labels)
        g_loss = self.criterion(outputs.squeeze(), real_labels)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log('d_loss', d_loss, prog_bar=True, on_epoch=True)
        self.log('g_loss', g_loss, prog_bar=True, on_epoch=True)

        return g_loss

    def on_train_epoch_end(self):
        # Generate images for each class to visualize cGAN performance
        n_samples = 10
        z = torch.randn(n_samples, latent_dim, device=self.device)
        labels = torch.arange(0, n_samples, device=self.device)
        generated_images = self.generator(z, labels).view(n_samples, 28, 28).detach().cpu().numpy()

        plt.figure(figsize=(10, 2))
        for i in range(n_samples):
            plt.subplot(1, n_samples, i + 1)
            plt.imshow(generated_images[i], cmap='gray')
            plt.axis('off')
            plt.title(str(i))
        plt.show(block=True)


# Set up TensorBoard logger
tensorboard_logger = TensorBoardLogger("lightning_logs", name="cGAN")

# Set up model checkpointing
checkpoint_callback = ModelCheckpoint(
    monitor="g_loss",
    dirpath="checkpoints",
    filename="cgan-best-model",
    save_top_k=1,
    mode="min",
    verbose=True
)

#Set to gpu if available

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Train the Conditional GAN
model = GAN()
trainer = L.Trainer(
    max_epochs=epochs,
    accelerator="gpu",
    logger=tensorboard_logger,
    callbacks=[checkpoint_callback],
    log_every_n_steps=1
)
trainer.fit(model, train_loader)


#Lets Inference the model

model.eval()
device = next(model.parameters()).device

num_samples = 5  # how many images you want to generate
class_id = int(input('Please input your class id:'))    # the digit you want to generate

z = torch.randn(num_samples, latent_dim, device=device)
labels = torch.full((num_samples,), class_id, dtype=torch.long, device=device)

with torch.no_grad():
    generated_imgs = model.generator(z, labels).cpu()

# Plot or save generated_imgs as needed
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 2))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(generated_imgs[i].view(28, 28), cmap='gray')
    plt.axis('off')
    plt.title(str(class_id))
plt.show()