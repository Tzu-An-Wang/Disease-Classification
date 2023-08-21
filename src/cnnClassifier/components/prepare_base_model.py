import numpy as np
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class PrepareBaseModel():
    def __init__(self, config:PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None


    def get_base_model(self):
        self.model = models.vgg16(
            pretrained=(self.config.params_weights)
        )
        if not self.config.params_include_top:
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.save_model(self.config.base_model_path, self.model)


    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif (freeze_till is not None) and (freeze_till>0):
            ct = 0
            for child in model.child:
                ct += 1
                if ct < freeze_till:
                    for param in model.parameters():
                        param.requires_grad = False
        
        num_flat_features = lambda x: int(np.prod(x.size()[1:]))
        with torch.no_grad():
            sample_tensor = torch.randn(1, 3, 224, 224)
            out = model(sample_tensor)
            in_feature = num_flat_features(out)

        full_model = nn.Sequential(
            model,
            nn.Flatten(),
            nn.Linear(in_features=in_feature, out_features=classes),
            nn.Softmax(dim=1)
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(full_model.parameters(), lr=learning_rate)
        print(full_model)

        return full_model, criterion, optimizer
    
    def update_base_model(self):
        self.full_model, criterion, optimizer = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes, 
            freeze_all=True, 
            freeze_till=None, 
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path:Path, model:nn.Module):
        torch.save(model.state_dict(), path)