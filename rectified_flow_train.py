import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torchvision.utils as vutils
from unet import ConditionalUNet  


seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class EMAHelper:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        ms = model.state_dict()
        for k, v in ms.items():
            v_cpu = v.detach().cpu()
            self.shadow[k].mul_(self.decay).add_(v_cpu, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone().cpu() for k, v in sd.items()}

    def copy_to(self, model):
        sd = {k: v.to(next(model.parameters()).device) for k, v in self.shadow.items()}
        model.load_state_dict(sd)


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])


class CelebADataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff"))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img 


def get_dataloader(root, batch_size=32, num_workers=4, pin_memory=True):
    dataset = CelebADataset(root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=True)


def sample_xt(x0, t):
   
    noise = torch.randn_like(x0)
    xt = (1.0 - t) * x0 + t * noise
    return xt, noise


def velocity_target(x0, noise):
    
    return noise - x0


@torch.no_grad()
def rk4_step(f, x, t, dt):
    k1 = f(x, t)
    k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(x + dt * k3, t + dt)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# -----------------------
# Sampling function
# -----------------------
@torch.no_grad()
def sample_from_checkpoint(checkpoint_path, out_path="samples/sample.png", steps=2000, batch_size=1):

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "ema_state" in ckpt:
        ema_state = ckpt["ema_state"]
    else:
        ema_state = ckpt

    model = ConditionalUNet(in_channels=3, out_channels=3, base_channels=64, time_emb_dim=512).to(device)
    ema = EMAHelper(model)
    ema.load_state_dict(ema_state)
    ema.copy_to(model)
    model.eval()

    x = torch.randn(batch_size, 3, 128, 128, device=device)
    dt = -1.0 / steps
    t = 1.0

    def velocity_fn(x_in, t_scalar):
        t_tensor = torch.full((x_in.size(0),), t_scalar, device=x_in.device)
        return model(x_in, t_tensor)

    for _ in range(steps):
        x = rk4_step(velocity_fn, x, t, dt)
        t += dt

    x = x.clamp(-1, 1)
    x = (x + 1.0) * 0.5 
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vutils.save_image(x, out_path, nrow=batch_size)
    print("Saved sample to:", out_path)


# -----------------------
# Training function
# -----------------------
def train_rectified_flow(
        data_root="img_align_celeba",
        epochs=100,
        batch_size=64,
        lr=1e-5,
        save_path="checkpoints/rf_best.pth",
        num_workers=4,
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    dataloader = get_dataloader(data_root, batch_size=batch_size, num_workers=num_workers)

    n_gpus = torch.cuda.device_count()
    device_ids = list(range(n_gpus))
    model = ConditionalUNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        time_emb_dim=512
    )
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)


    ema = EMAHelper(model, decay=0.999)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for x0 in pbar:
            x0 = x0.to(device)
            B = x0.size(0)
            global_step += 1

            t_1d = torch.rand(B, device=device)         
            t = t_1d.view(B, 1, 1, 1)                   

            xt, noise = sample_xt(x0, t)
            v_target = velocity_target(x0, noise)
           
            v_pred = model(xt, t_1d)

            loss = loss_fn(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ema.update(model)

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg:.6f}")

        
        ckpt = {
            "epoch": epoch + 1,
            "ema_state": ema.state_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg
        }
        torch.save(ckpt, f"checkpoints/epoch_{epoch+1}.pth")

        if avg < best_loss:
            best_loss = avg
            torch.save(ckpt, save_path)
            print(f"✔ Saved best model (epoch {epoch+1}) loss={best_loss:.6f}")

    print("Training complete.")


if __name__ == "__main__":
    
    DATA_ROOT = "img_align_celeba"  
    EPOCHS = 1000
    BATCH = 64
    LR = 1e-5                
    CHECKPOINT = "checkpoints/rf_best.pth"
    NUM_WORKERS = 4

    train_rectified_flow(
        data_root=DATA_ROOT,
        epochs=EPOCHS,
        batch_size=BATCH,
        lr=LR,
        save_path=CHECKPOINT,
        num_workers=NUM_WORKERS
    )

    # Generate Samples after training
    # sample_from_checkpoint(CHECKPOINT, out_path="samples/sample.png", steps=2000, batch_size=4)
