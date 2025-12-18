from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from unet import ConditionalUNet
from tensorflow.keras.datasets import mnist


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),    
    transforms.Normalize((0.5,), (0.5,))
])


class MNISTKerasDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx] 

        if self.transform:
            img = self.transform(img)

        return img


(x_train, _), (_, _) = mnist.load_data()

dataset = MNISTKerasDataset(x_train, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device set: {device}')
losses = []



# -------------------------------
# Utility Functions for Flow Matching
# -------------------------------
def sample_xt(x0, x1, t):
    return (1 - t) * x0 + t * x1

def velocity_target(x0, x1):
    return x1 - x0



# --------------------------------
# RK4 ODE Integrator for Sampling
# --------------------------------
@torch.no_grad()
def rk4_step(f, x, t, dt):
    k1 = f(x, t)
    k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(x + dt * k3, t + dt)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)



# -------------------------------
# Inference Function to Generate Samples
# -------------------------------
@torch.no_grad()
def generate_sample(model, img_size=(3,128,128), steps=100):
    model.eval()
    C, H, W = img_size
    x = torch.randn(1, C, H, W).to(device)
    dt = -1.0 / steps
    t = 1.0

    def velocity_fn(x_in, t_scalar):
        t_tensor = torch.full((x_in.size(0),), t_scalar, device=x_in.device)
        return model(x_in, t_tensor)

    for _ in range(steps):
        x = rk4_step(velocity_fn, x, t, dt)
        t += dt

    return x.clamp(-1, 1)



# -------------------------------
# Training Function
# -------------------------------
def train_flow_matching(model, dataloader, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x0 in dataloader:
            x0 = x0.to(device)

            x1 = torch.randn_like(x0)

            t = torch.rand(x0.size(0), device=device).view(-1, 1, 1, 1)

            xt = sample_xt(x0, x1, t)
            v_target = velocity_target(x0, x1)

            v_pred = model(xt, t.view(-1))

            loss = loss_fn(v_pred, v_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss / len(dataloader):.6f}")
        losses.append(epoch_loss / len(dataloader))

    return losses




# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    device_ids = list(range(n_gpus))

    model = ConditionalUNet()
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    print("Starting flow matching training on MNIST...")
    train_flow_matching(model, dataloader, epochs=20, lr=1e-4)

    torch.save(model.state_dict(), "flow_matching_mnist.pth")
    print("Training complete.")
