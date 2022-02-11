from torchvision import transforms
from PyTorch_Pretrained_ViT.pytorch_pretrained_vit.model import ViT
import json
import torch
import datetime
from PIL import Image

model_name = 'B_32_imagenet1k'
model = ViT(model_name, pretrained=True)

img = Image.open("PyTorch_Pretrained_ViT/examples/simple/img.jpg")

tfms = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(),
                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
img = tfms(img).unsqueeze(0)


# Load class names
labels_map = json.load(open('PyTorch_Pretrained_ViT/examples/simple/labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

model.eval()
with torch.no_grad():
    tik = datetime.datetime.now()
    outputs, feature_maps = model(img)
    output = outputs[0].squeeze(0)
    print(datetime.datetime.now() - tik)
print('-----')
for idx in torch.topk(output, k=3).indices.tolist():
    prob = torch.softmax(output, -1)[idx].item()
    print('[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100))

