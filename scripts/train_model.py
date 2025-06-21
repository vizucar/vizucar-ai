import os
import glob
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# === CONFIGURATION ===
DATA_DIR = os.getenv("DATA_DIR", "data/processed")
PROMPT_FILE = os.getenv("PROMPT_FILE", "data/prompts/prompts.json")
OUTPUT_DIR = "models/custom_unet"
GENERATED_DIR = "models/outputs"
PRETRAINED_MODEL = "openai/clip-vit-base-patch32"
PIPELINE_MODEL = "CompVis/stable-diffusion-v1-4"
EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
SAVE_EVERY = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# === LOAD TOKENIZER + TEXT ENCODER ===
tokenizer = CLIPTokenizer.from_pretrained(PIPELINE_MODEL, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PIPELINE_MODEL, subfolder="text_encoder").to(DEVICE)

# === LOAD PRETRAINED PIPELINE ===
pipe = StableDiffusionPipeline.from_pretrained(PIPELINE_MODEL, torch_dtype=torch.float32).to(DEVICE)
unet = pipe.unet
vae = pipe.vae

# === FREEZE COMPONENTS (optional) ===
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# === IMAGE TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === CUSTOM DATASET ===
class CarDataset(Dataset):
    def __init__(self, images_dir, prompts_file, transform):
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []

        with open(prompts_file, 'r') as f:
            self.samples = json.load(f) 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = os.path.join(self.images_dir, item["image_path"])
        prompt = item["prompt_general"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return {"pixel_values": image, "prompt": prompt}

dataset = CarDataset(DATA_DIR, PROMPT_FILE, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === SCHEDULER ===
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# === ACCELERATOR ===
accelerator = Accelerator()
optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

# === CHARGER CHECKPOINT SI EXISTE ===
def load_checkpoint(unet, optimizer, output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "unet_epoch_*.pt"))
    if not checkpoints:
        print("No checkpoints found, starting from scratch.")
        return unet, optimizer, 0
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    print(f"Chargement du checkpoint : {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt)
    unet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    return unet, optimizer, start_epoch

unet, optimizer, start_epoch = load_checkpoint(unet, optimizer, OUTPUT_DIR)

# === TRAINING LOOP ===
for epoch in range(start_epoch, EPOCHS):
    unet.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = batch["pixel_values"].to(DEVICE)
        prompts = batch["prompt"]

        # Tokenize prompts
        input_ids = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(DEVICE)
        prompt_embeds = text_encoder(input_ids)[0]

        # Encode images to latent space
        with torch.no_grad():
            latents = vae.encode(images * 2 - 1).latent_dist.sample() * 0.18215

        # Sample noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    with open("logs/train.log", "a") as f:
        f.write(f"{epoch+1},{avg_loss:.6f}\n")

    # === SAVE UNET CHECKPOINT ===
    if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == EPOCHS:
        unwrapped = accelerator.unwrap_model(unet)
        ckpt_path = os.path.join(OUTPUT_DIR, f"unet_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': unwrapped.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)
        print(f"📦 Saved model checkpoint: {ckpt_path}")

        # === TEST IMAGE GENERATION ===
        unet.eval()
        test_prompt = "a red sports car on a racetrack"
        prompt_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            prompt_embeds = text_encoder(prompt_ids)[0]

        pipe.unet = unwrapped
        image = pipe(prompt=test_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

        gen_dir = os.path.join(GENERATED_DIR, f"epoch_{epoch+1}")
        os.makedirs(gen_dir, exist_ok=True)
        image_path = os.path.join(gen_dir, "test_prompt.jpg")
        image.save(image_path)
        print(f"🖼️ Generated test image: {image_path}")