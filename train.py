import torch
import torch.nn as nn
import torch.nn.functional as F

from model import *
from dataPreparation import *

class ConLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.temp = 0.1

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            mask = mask ^ torch.diag_embed(torch.diag(mask))

        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp

        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))

        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()

        exp_logits = torch.exp(logits)

        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        return loss

    def forward(self, output, targets):
        normed_output = F.normalize(output, dim=-1)
        cl_loss = self.nt_xent_loss(normed_output, normed_output, targets)
        return cl_loss
    

epochs = 15
min_val_loss = 1e5
patience = 3

attention_model = attention_model()
classifier = classifier()

optimizer1 = torch.optim.Adam(attention_model.parameters(), lr=5e-5)
optimizer2 = torch.optim.Adam(classifier.parameters(), lr=5e-5)

contrastiveLoss = ConLoss()
entropyLoss = torch.nn.CrossEntropyLoss()

count = 0
for i in range(epochs):
    attention_model.train()

    for data in train_loader:
        attention_model.zero_grad()

        text_embeds = data["text_embeds"]
        images_x = data["images_x"]
        violent_label = data["violent_label"]
        real_label = data["real_label"]
        senti_label = data["sentiment_label"]

        combined_features = attention_model(images_x, text_embeds)

        loss = contrastiveLoss.forward(output=combined_features, targets=real_label) + contrastiveLoss.forward(output=combined_features, targets=violent_label) + contrastiveLoss.forward(output=combined_features, targets=senti_label)

        loss.backward()
        optimizer1.step()

    attention_model.eval()
    with torch.no_grad():
        total_loss_val = 0
        total_val = 0

        for data_val in val_loader:

            text_embeds_val = data_val["text_embeds"]
            images_x_val = data_val["images_x"]
            violent_val_label = data_val["violent_label"]
            real_val_label = data_val["real_label"]
            senti_val_label = data_val["sentiment_label"]

            combined_features_val = attention_model(images_x_val, text_embeds_val)

            loss_val = contrastiveLoss(output=combined_features_val, targets=real_val_label) + contrastiveLoss(output=combined_features_val, targets=senti_val_label) + contrastiveLoss(output=combined_features_val, targets=senti_val_label)

            total_val += text_embeds_val.size(0)
            total_loss_val += loss_val.item()
            
        val_loss = total_loss_val / total_val

        if (min_val_loss > val_loss):
            min_val_loss = val_loss
        else:
            count += 1
            if count == patience:
                break

    print(f"Epoch : {i+1}")
    print(f"Training Loss: {loss.item()}")
    print(f"Validation loss : {val_loss}")

count = 0
for i in range(epochs):
    classifier.train()

    for data in train_loader:
        classifier.zero_grad()

        text_embeds = data["text_embeds"]
        images_x = data["images_x"]
        violent_label = data["violent_label"]
        real_label = data["real_label"]
        senti_label = data["sentiment_label"]

        combined_features = attention_model(images_x, text_embeds)

        real, violent, senti = classifier(combined_features)

        loss = entropyLoss(real, real_label) + entropyLoss(violent, violent_label) + entropyLoss(senti, senti_label)

        loss.backward()
        optimizer2.step()

    classifier.eval()
    with torch.no_grad():
        total_loss_val = 0
        total_val = 0

        for data_val in val_loader:

            text_embeds_val = data_val["text_embeds"]
            images_x_val = data_val["images_x"]
            violent_val_label = data_val["violent_label"]
            real_val_label = data_val["real_label"]
            senti_val_label = data_val["sentiment_label"]

            combined_features_val = attention_model(images_x_val, text_embeds_val)

            real_val, violent_val, senti_val = classifier(combined_features_val)

            loss_val = entropyLoss(real_val, real_val_label) + entropyLoss(violent_val, violent_val_label) + entropyLoss(senti_val, senti_val_label)

            total_val += text_embeds_val.size(0)
            total_loss_val += loss_val.item()

        val_loss = total_loss_val / total_val

        if (min_val_loss > val_loss):
            min_val_loss = val_loss
        else:
            count += 1
            if count == patience:
                break

    print(f"Epoch : {i+1}")
    print(f"Training Loss: {loss.item()}")
    print(f"Validation loss : {val_loss}")
