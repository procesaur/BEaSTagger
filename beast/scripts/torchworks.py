import numpy as np
import pandas as pd
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        self.layer_out = nn.Linear(num_feature, num_class)

    def forward(self, x):
        x = self.layer_out(x)
        return x


class ClassifierDatasetx(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def test_prob_net(csv, par_path, out_path, modelname='net-prob.pt'):

    dataset = pd.read_csv(csv, sep='\t')
    modelnameb = modelname.split(".pt")[0]+".pt"

    with open(par_path + '/' + modelnameb + '.col', 'rb') as p:
        colnames = json.load(p)

    colnames = list(col for col in colnames if col != 'result')

    dataset = dataset.reindex(columns=colnames)
    dataset = dataset.fillna(0)

    input = dataset.iloc[:, dataset.columns != 'result']
    output = dataset.iloc[:, 0]

    input, output = np.array(input), np.array(output)

    with open(par_path + '/' + modelnameb + '.p', 'rb') as p:
        class2idx = json.load(p)
    idx2class = {v: k for k, v in class2idx.items()}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MulticlassClassification(num_feature=len(colnames), num_class=len(class2idx))
    model.load_state_dict(torch.load(par_path + '/' + modelname, map_location=torch.device(device)))
    model.to(device)

    test_dataset = ClassifierDatasetx(torch.from_numpy(input).float(), torch.from_numpy(output).long())

    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    y_pred_list = []
    y_prob_list = []

    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_softmax = torch.log_softmax(y_test_pred, dim=1)
            sm = nn.Softmax(dim=1)
            probs = sm(y_test_pred)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
            y_prob_list.append(torch.max(probs).item())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    tags = ([])
    for y in y_pred_list:
        tags.append(idx2class[y])

    return tags, y_prob_list


def train_prob_net(csv, out, name, epochs=100, batch_size=32, lr=0.001, val_size=0.1):

    out_path = out + "/" + name
    dataset = pd.read_csv(csv, sep='\t')
    nodedict = {}
    for c in dataset.columns:
        if c != 'result':
            if c.split('__')[1] not in nodedict.keys():
                nodedict[c.split('__')[1]] = ([])

    for k in nodedict:
        for i, c in enumerate(dataset.columns):
            if c != 'result':
                if c.split('__')[1] == k:
                    nodedict[k].append(i - 1)

    output = dataset.iloc[:, 0]
    tags = output.tolist()

    for i, tag in enumerate(tags):
        tags[i] = tag.split('\t')[0]

    tag_list = list(set(tags))

    if os.path.isfile(out_path + '.p'):
        with open(out_path + '.p', 'rb') as p:
            class2idx = json.load(p)
        idx2class = {v: k for k, v in class2idx.items()}
    else:
        print('classmap not found : ' + out_path + '.p')
        class2idx = {}
        for i, key in enumerate(tag_list):
            class2idx[key] = int(i)
        idx2class = {v: k for k, v in class2idx.items()}

    dataset['result'].replace(class2idx, inplace=True)

    input = dataset.iloc[:, dataset.columns != 'result']
    output = dataset.iloc[:, 0]

    # print(dataset.head)

    for i in range(30):
        try:
            X_trainval, X_test, y_trainval, y_test = train_test_split(input, output, test_size=0.1)

            X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_size,
                                                              stratify=y_trainval)
            break
        except ValueError as e:
            print(str(e) + " ...trying again")

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    train_dataset = ClassifierDatasetx(torch.tensor(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDatasetx(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDatasetx(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]

    def get_class_distribution(obj):
        count_dict = {}
        for i, t in enumerate(tag_list):
            count_dict[i] = 0

        for i in obj:
            count_dict[i] += 1

        return count_dict

    class_count = [i for i in get_class_distribution(y_train).values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)

    # print(class_weights)

    class_weights_all = class_weights[target_list]

    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    EPOCHS = epochs
    BATCH_SIZE = batch_size
    LEARNING_RATE = lr
    NUM_FEATURES = len(input.columns)
    NUM_CLASSES = len(tag_list)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=weighted_sampler
                              )
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = MulticlassClassification2(num_feature=NUM_FEATURES, num_class=NUM_CLASSES, nodedict=nodedict)
    model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # optimizer = torch.optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # print(model)

    def multi_acc(y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)

        acc = torch.round(acc) * 100

        return acc

    accuracy_stats = {
        'train': [],
        "val": [],
    }
    loss_stats = {
        'train': [],
        "val": [],
    }

    print("Begin training.")
    best = 0
    lastbest = 0

    for e in range(1, EPOCHS + 1):

        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()

            def closure():
                return train_loss

            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        tloss = train_epoch_loss / len(train_loader)
        vloss = val_epoch_loss / len(val_loader)
        tacc = train_epoch_acc / len(train_loader)
        vacc = val_epoch_acc / len(val_loader)

        print(
            f'Epoch {e + 0:03}: | Train Loss: {tloss:.5f} | Val Loss: {vloss:.5f} | Train Acc: {tacc:.3f}| Val Acc: {vacc:.3f}')
        score = vacc
        lastbest = 0
        if best/score < 1.002 and tloss+vloss < 0.064 and tloss/vloss > 0.9:
            if score > best:
                best = score
            lastbest = 0
            torch.save(model.state_dict(), out_path + '_' + str(score))
            torch.save(model.state_dict(), out_path)
        else:
            lastbest += 1

    torch.save(model.state_dict(), out_path)

    with open(out_path + '.p', 'w', encoding="utf-8") as fp:
        json.dump(class2idx, fp)

    colnames = ([])
    for col in dataset.columns:
        colnames.append(col)

    with open(out_path + '.p', 'w', encoding="utf-8") as fp:
        json.dump(class2idx, fp)

    with open(out_path + '.col', 'w', encoding="utf-8") as fp:
        json.dump(colnames, fp)

    y_pred_list = []

    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_pred_softmax = torch.log_softmax(y_test_pred, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    print(classification_report(y_test, y_pred_list, zero_division=1))
