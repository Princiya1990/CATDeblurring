import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim import Adam
from PIL import Image

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual

class ImageTranslatorGenerator(nn.Module):
    def __init__(self):
        super(ImageTranslatorGenerator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(9)])

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.output(x)

# Discriminator Architecture
class ImageTranslatorDiscriminator(nn.Module):
    def __init__(self):
        super(ImageTranslatorDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        return self.fc(x).view(-1)

class ImageDataset(Dataset):
    def __init__(self, sketch_dir, image_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.image_dir = image_dir
        self.transform = transform
        self.sketches = os.listdir(sketch_dir)

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketch_dir, self.sketches[idx])
        image_path = os.path.join(self.image_dir, self.sketches[idx])

        sketch = Image.open(sketch_path).convert("RGB")
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            sketch = self.transform(sketch)
            image = self.transform(image)

        return sketch, image

def train(generator, discriminator, dataloader, optimizer_g, optimizer_d, epochs, device):
    loss_function = nn.BCEWithLogitsLoss()
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    for epoch in range(epochs):
        for i, (sketches, images) in enumerate(dataloader):
            sketches, images = sketches.to(device), images.to(device)

            # Train Discriminator
            optimizer_d.zero_grad()
            real_preds = discriminator(images)
            fake_images = generator(sketches)
            fake_preds = discriminator(fake_images.detach())

            real_loss = loss_function(real_preds, torch.ones_like(real_preds))
            fake_loss = loss_function(fake_preds, torch.zeros_like(fake_preds))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_preds = discriminator(fake_images)
            g_loss = loss_function(fake_preds, torch.ones_like(fake_preds))
            g_loss.backward()
            optimizer_g.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

def save_generated_images(generator, sketch_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    sketches = os.listdir(sketch_dir)

    generator = generator.to(device)
    generator.eval()

    for sketch_name in sketches:
        sketch_path = os.path.join(sketch_dir, sketch_name)
        sketch = Image.open(sketch_path).convert("RGB")
        sketch = transform(sketch).unsqueeze(0).to(device)

        with torch.no_grad():
            generated_image = generator(sketch).squeeze(0).cpu()
            save_image(generated_image, os.path.join(output_dir, f"generated_{sketch_name}"))

# Paths
sketch_dir = "sketches"
image_dir = "images"
output_dir = "generated_images"


epochs = 500
batch_size = 16
learning_rate = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = ImageDataset(sketch_dir, image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
generator = ImageTranslatorGenerator()
discriminator = ImageTranslatorDiscriminator()
optimizer_g = Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_d = Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Train Model
train(generator, discriminator, dataloader, optimizer_g, optimizer_d, epochs, device)

save_generated_images(generator, sketch_dir, output_dir, device)
print(f"Generated images saved in {output_dir}")
