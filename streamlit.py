import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.transforms import Normalize
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

class Generator(nn.Module):
  def __init__(self, z_dim=100, image_size=28*28, num_classes=10):
    super().__init__()
    self.label_emb = nn.Embedding(num_classes, num_classes)
    self.model = nn.Sequential(
        nn.Linear(z_dim + num_classes, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, image_size),
        nn.Tanh())

  def forward(self, z, labels):
    c = self.label_emb(labels)
    x = torch.cat([z, c], 1)
    return self.model(x)
  
def generate_images(digit, n_images=5):
	z_dim = 100
	image_size = 28*28
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
	generator = Generator(z_dim, image_size, 10).to(device)
	generator.load_state_dict(torch.load("generator.pth", map_location=device))
  generator.eval()
	z = torch.randn(n_images, z_dim)
	labels = torch.full((n_images,), digit, dtype=torch.long)
	with torch.no_grad():
		generated = generator(z, labels).view(-1, 1, 28, 28)
	  generated = (generated + 1) / 2  # Rescale to [0, 1]
	return generated

def display_images(images):
		grid = make_grid(images, nrow=5, normalize=True, range=(0, 1))
		np_image = grid.cpu().numpy().transpose((1, 2, 0)) * 255
		np_image = np.clip(np_image, 0, 255).astype(np.uint8)
		
		fig, ax = plt.subplots(figsize=(5, 5))
		ax.imshow(np_image)
		ax.axis('off')
		
		buf = io.BytesIO()
		plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
		buf.seek(0)
		
		return buf

st.set_page_config(page_title="MNIST Digit Generator", layout="wide")
st.title("MNIST Digit Generator by Sayandeep Dey")
digit = st.selectbox("Select a digit (0-9):", list(range(10)))

if st.button("Generate Images"):
		n_images = st.slider("Number of images to generate:", 1, 10, 5)
		with st.spinner("Generating images..."):
				images = generate_images(digit, n_images)
				image_buf = display_images(images)
				st.image(image_buf, caption=f"Generated images for digit {digit}", use_column_width=True)
