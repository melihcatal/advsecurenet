import torch
import torchvision.models as models
import torchvision.transforms as transforms
import json
import pkg_resources
from PIL import Image
import os 
class VGG16Features(torch.nn.Module):
    def __init__(self):
        super(VGG16Features, self).__init__()
        
        vgg_pretrained = models.vgg16(pretrained=True)
        self.features = vgg_pretrained.features
        self.avgpool = vgg_pretrained.avgpool
        self.classifier = vgg_pretrained.classifier

        # Load the ImageNet class labels using pkg_resources
        resource_package = 'advsecurenet'
        resource_path = os.path.join('resources', 'imagenet_class_index.json')

        json_content = pkg_resources.resource_string(resource_package, resource_path)
        self.class_idx = json.loads(json_content.decode('utf-8'))

        # Define the transformation pipeline for the images
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i == 5:  # this is the FC7 layer
                features = x
        logits = x
        return logits, features
    
    def predict_class(self, input_tensor):
        logits, _ = self.forward(input_tensor)
        _, predicted_class = torch.max(logits, 1)
        idx2label = [self.class_idx[str(k)][1] for k in range(len(self.class_idx))]
        return idx2label[predicted_class.item()]

    def preprocess_image(self, image):
        return self.preprocess(image).unsqueeze(0)
