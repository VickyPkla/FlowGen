import torch
import matplotlib.pyplot as plt
from unet import ConditionalUNet
from mnist_train import generate_sample 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


model = ConditionalUNet()
ckpt = torch.load("load/model/path.pth", map_location=device)

if "model_state_dict" in ckpt:
    ckpt = ckpt["model_state_dict"]

new_state_dict = {}
for key, val in ckpt.items():
    new_key = key.replace("module.", "")
    new_state_dict[new_key] = val

model.load_state_dict(new_state_dict, strict=True)
model = model.to(device)
model.eval()


print("Generating new sample...")
sample = generate_sample(model, img_size=(1, 128, 128), steps=100)

# Post-process image
img = sample[0].cpu()
img = (img * 0.5 + 0.5).clamp(0, 1)  
img = img.permute(1, 2, 0).numpy()        

plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.axis("off")
plt.title("Generated Sample")
plt.show()
