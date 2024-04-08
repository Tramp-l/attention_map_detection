from attention_extraction import extract_attention_weights
from data_process import preprocess_data
from ssim import calculate_ssim
from torchvision.datasets import ImageNet
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from transformers import ViTModel

model = ViTModel.from_pretrained('./dino_imagenet2012_ViT-B_12-224.pth')

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

dataset = ImageNet(root='./data', split='val', transform=transform, download=True)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

for images, labels in data_loader:

    processed_data = preprocess_data(images)

    attention_mid, attention_final = extract_attention_weights(model, images)

    ssim = calculate_ssim(attention_mid, attention_final)

    threshold = 0.6

    if ssim < threshold:
        print('对抗样本')
    else:
        print('正常样本')
