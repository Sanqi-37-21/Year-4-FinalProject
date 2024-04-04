import torch
import torchvision
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time



device = "cuda" if torch.cuda.is_available() else "cpu"

#beta increase from begin to end
def beta_increase(steps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, steps)

#calculate Xt at any time
def X_T(at1, X0, at2, zt):
    return at1*X0 + at2*zt

#forward diffusion
def forward_step(x0,t):
    noise = torch.randn_like(x0)
    sqrt_alpha_cumprod_t = extract(sqrt_alpha_cumprod, t, x0.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(
        sqrt_one_minus_alpha_cumprod, t, x0.shape
    )
    #print('sqrt_one_minus_alpha_cumprod_t',sqrt_one_minus_alpha_cumprod_t.shape,'noise',noise.shape)
    noised_img = sqrt_alpha_cumprod_t.to(device) * x0.to(device) + sqrt_one_minus_alpha_cumprod_t.to(device) * noise.to(device)
    #print('noised_img',noised_img.shape)
    return noised_img ,noise.to(device)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.to(device))
    # print(out)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)


#number of steps
steps = 500

#initial beta
beta = beta_increase(steps)
#define alpha
alpha = 1-beta
#cumprod of alpha
alpha_cumprod = torch.cumprod(alpha.to(device), axis = 0)
#sqrt(bar(a))
sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
#sqrt(1-bar(a))
sqrt_one_minus_alpha_cumprod = torch.sqrt(1-alpha_cumprod)
#1/sqrt(a)
one_over_sqrt_alpha = torch.sqrt(1.0 / alpha)
#b/sqrt(1-bar(a))
beta_over_sqrt_one_minus_alpha_cumprod= beta/sqrt_one_minus_alpha_cumprod.to('cpu')

posterior_variance = beta.to(device) * (1. - alpha_cumprod_prev.to(device)) / (1. - alpha_cumprod.to(device))

#reform the picture
img_size = 64
batch_size = 64
train_path = "./front"
train_path2 = "./Abstract_gallery_2"

def load_trans_dataset():
    data_trans = [
        transforms.Resize((img_size,img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t:(t*2)-1)
    ]

    tran = transforms.Compose(data_trans)
    train = torchvision.datasets.ImageFolder(root=train_path, transform=tran)
    train2 = torchvision.datasets.ImageFolder(root=train_path2, transform=tran)
    #this will combin the datasets together
    return torch.utils.data.ConcatDataset([train,train2])

def reverse_trans(image):
    reverse_trans = transforms.Compose([
        transforms.Lambda(lambda t: (t+1)/2),
        transforms.Lambda(lambda t: t.permute(1,2,0)),
        transforms.Lambda(lambda t: t*255),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        #transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    image = reverse_trans(image)
    # print('img this is img after trans wow', image)
    plt.imshow(image)

data = load_trans_dataset()
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

from torch import nn
import math


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()



    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # print('shape x', x.shape)
        # print('shape t', t.shape)
        # print('shape h', h.shape)
        # print('shape time_emb', time_emb.shape)
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # print('shape time_emb', time_emb.shape)
        #print('hhhhhhhhhhhhh',h.shape)
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], \
                                          time_emb_dim) \
                                    for i in range(len(down_channels) - 1)])
        # Upsample99
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], \
                                        time_emb_dim, up=True) \
                                  for i in range(len(up_channels) - 1)])

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # print('thisis', x.shape)
        # Unet
        residual_inputs = []
        for down in self.downs:
            # print('this is down', down)
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        #print('shit is is ',x.shape)
        #print(self.output(x).shape)
        return self.output(x)


model = SimpleUnet().to(device)
# next two line is for load the pretrained data into model
# m_state_dict = torch.load('1000.pt')
# model.load_state_dict(m_state_dict)
print("Num params: ", sum(p.numel() for p in model.parameters()))

def get_loss(model, x_0, t):
    x_noisy, noise = forward_step(x_0, t)
    noise_pred = model(x_noisy, t)
    #print('noise shape', noise.shape)
    #print('noise_pred shape', noise_pred.shape)
    return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = extract(beta.to(device), t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alpha_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(one_over_sqrt_alpha.to(device), t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = extract(posterior_variance, t, x.shape)
    #print('reverse_transform img 33333', posterior_variance)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        a = torch.sqrt(posterior_variance_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(milestone):
    # Sample noise
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 4
    stepsize = int(steps / num_images)

    for i in range(0, steps)[::-1]:
        t = torch.full((1,), i, device='cuda', dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        #print('img this is img wow', img.shape)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            reverse_trans(img.detach())
    plt.savefig(f'1000output/out-{milestone}.png', dpi=300)
    plt.close()

from torch.optim import Adam

optimizer = Adam(model.parameters(), lr=0.001)
#epochs = 5 # Try more!
epochs = [50,100,150,200,250,300,350,400,450,500] # Try more!

timelsit = []
for epoch in range(1,501):
    time_start = time.time()
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()
      t = torch.randint(0, steps, (batch_size,), device=device).long()
      #print(get_loss(model, batch[0], t))
      loss = get_loss(model, batch[0], t)
      loss.backward()
      optimizer.step()
      if epoch % 1 == 0 and step == 0:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        sample_plot_image(epoch)

    if epoch in epochs:
        # save trained data from the model
        torch.save(model.state_dict(), f'{epoch}_LittleCharacter.pt')

        time_end = time.time()
        time_sum = time_end - time_start
        print(time_sum)
        timelsit.append(time_sum)

print(timelsit)

