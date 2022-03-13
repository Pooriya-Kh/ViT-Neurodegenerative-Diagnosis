import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def cnn_report(model, dataloader, device):
    accs = []
    losses = []
    y_true = []
    y_pred = []
    
    # criterion = nn.CrossEntropyLoss()
    label2id = {"CN": 0, "MCI": 1, "AD": 2}
    
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            x, y  = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            # loss = criterion(logits, y)
            # accuracy = ((preds == y).sum()) / logits.shape[0]
            # losses.append(loss.item())
            # accs.append(accuracy.item())
            y_pred.append(preds.cpu().numpy())
            y_true.append(y.cpu().numpy())
    
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    print(classification_report(y_true, y_pred, labels=[0, 1, 2],  target_names=list(label2id.keys())))
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label2id.keys()))
    disp.plot(xticks_rotation=45);


def vit_report(model, feature_extractor, dataloader, device):
    accs = []
    losses = []
    y_true = []
    y_pred = []
    
    criterion = nn.CrossEntropyLoss()
    label2id = {"CN": 0, "MCI": 1, "AD": 2}
    
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            x = np.split(np.array(x), dataloader.batch_size)
            for i in range(len(x)):
                x[i] = np.squeeze(x[i])
            x = torch.tensor(np.stack(feature_extractor(x)['pixel_values'], axis=0))
            x, y  = x.to(device), y.to(device)
            logits, _ = model(x)
            preds = logits.argmax(1)
            # loss = criterion(logits, y)
            # accuracy = ((preds == y).sum()) / logits.shape[0]
            # losses.append(loss.item())
            # accs.append(accuracy.item())
            y_pred.append(preds.cpu().numpy())
            y_true.append(y.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    print(classification_report(y_true, y_pred, labels=[0, 1, 2],  target_names=list(label2id.keys())))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label2id.keys()))
    disp.plot(xticks_rotation=45);