import torch
from torch import nn
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import ImageNet
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

teacher_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
student_model = ViTForImageClassification.from_pretrained('google/vit-small-patch16-224', num_labels=0).to(device)

for param in teacher_model.parameters():
    param.requires_grad = False

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

optimizer = AdamW(student_model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.001)

scheduler = CosineAnnealingLR(optimizer, T_max=300)

criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

dataset = ImageNet(root='path_to_imagenet', split='train', transform=transform)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for epoch in range(300):
    for images, _ in tqdm(dataloader):

        inputs = feature_extractor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        student_outputs = student_model(**inputs)

        loss = criterion(student_outputs.logits, teacher_outputs.logits.argmax(dim=-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    torch.save(student_model.state_dict(), "student_model.pth")