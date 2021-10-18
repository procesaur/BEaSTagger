import sys
import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(num_feature, num_class)

        self.relu = nn.ReLU()
        self.sm = nn.Softmax()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        # x = self.layer_1(x)
        # x = self.batchnorm1(x)
        # x = self.dropout1(x)
        # x = self.relu(x)

        # x = self.layer_2(x)
        # x = self.batchnorm2(x)
        # x = self.relu(x)


        # x = self.layer_3(x)
        # x = self.batchnorm3(x)
        # x = self.relu(x)
        # x = self.dropout2(x)

        x = self.layer_out(x)
        #x = self.sm(x)

        return x


class SingleTagModel(nn.Module):
    def __init__(self, n):
        super(SingleTagModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(n, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return x


class MulticlassClassification2(nn.Module):
    def __init__(self, num_feature, num_class, nodedict):
        super(MulticlassClassification2, self).__init__()

        ntags = len(nodedict)
        self.linears = nn.ModuleList([])
        self.list = ([])
        for d in nodedict:
            self.linears.append(SingleTagModel(len(nodedict[d])))
            self.list.append(nodedict[d])

        self.layer_1 = nn.Linear(num_feature, ntags)
        self.layer_out = nn.Linear(ntags, num_class)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(ntags)

    def forward(self, x):
        catarray = ([])
        for i, m in enumerate(self.linears):
            newt = torch.LongTensor(self.list[i])
            newx = x.index_select(1, newt)
            newx = self.relu(self.linears[i](newx))
            catarray.append(newx)

        x = torch.cat(catarray, 1)
        # x = self.layer_1(x)
        # x = self.batchnorm1(x)
        # x = self.relu(x)

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
        colnames = pickle.load(p)

    colnames = list(col for col in colnames if col != 'result')

    dataset = dataset.reindex(columns=colnames)
    dataset = dataset.fillna(0)

    input = dataset.iloc[:, dataset.columns != 'result']
    output = dataset.iloc[:, 0]

    input, output = np.array(input), np.array(output)

    with open(par_path + '/' + modelnameb + '.p', 'rb') as p:
        class2idx = pickle.load(p)
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


def train_prob_net(csv, out, name):

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
            class2idx = pickle.load(p)
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

    X_trainval, X_test, y_trainval, y_test = train_test_split(input, output, test_size=0.01,
                                                              random_state=66)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval,
                                                      random_state=20)

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

    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
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
    # criterion = nn.

    optimizer = torch.optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.005, lr_decay=0, weight_decay=0,
                                    initial_accumulator_value=0, eps=1e-10)
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

    with open(out_path + '.p', 'wb') as fp:
        pickle.dump(class2idx, fp, protocol=pickle.HIGHEST_PROTOCOL)

    colnames = ([])
    for col in dataset.columns:
        colnames.append(col)

    with open(out_path + '.p', 'wb') as fp:
        pickle.dump(class2idx, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out_path + '.col', 'wb') as fp:
        pickle.dump(colnames, fp, protocol=pickle.HIGHEST_PROTOCOL)

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
    print(classification_report(y_test, y_pred_list))


def probtagToMatrix(linest, taggershort, tags=None):
    thetags = ([])
    matrices = ([])
    fulltagset = ([])
    tagtrue = {}
    tagflase = {}
    tagaccu = {}

    truefpos = 0
    truepos = 0
    falsefpos = 0
    falsepos = 0

    ln = len(linest)
    tagger_tags = ([])

    for idx, line in enumerate(linest):
        line = line.rstrip('\n')
        resultsline = line.split('\t')
        top = resultsline[1]
        topn = float(top.split(' ')[1])
        toptag = top.split(' ')[0]
        word = resultsline[0]
        del resultsline[0]
        for nr in resultsline:
            thetags.append(taggershort + '__' + nr.split(' ')[0])

            try:
                istopn = float(nr.split(' ')[1])
            except:
                sys.stdout.write(line)
            if istopn > topn:
                topn = istopn
                toptag = nr.split(' ')[0]
                top = nr

        tagger_tags.append(toptag)

        if tags is not None:
            res = top.split(' ')[0]
            ans = tags[idx]

            if res not in tagtrue.keys():
                tagtrue[res] = 0
                tagflase[res] = 0

            if ans == res:
                truefpos += 1
                tagtrue[res] += 1
            else:
                falsefpos += 1
                tagflase[res] += 1

            if ans.split(':')[0] == res.split(':')[0]:
                truepos += 1
            else:
                falsepos += 1

            for key in tagtrue:
                tn = tagtrue[key]
                fn = tagflase[key]
                tagaccu[taggershort + '_' + key] = tn * 100 / (tn + fn)

    tagset = list(set(thetags))

    tagdic = {}
    for i, dt in enumerate(tagset):
        tagdic[dt] = i
        fulltagset.append(dt)

    matrix = np.zeros([len(tagset), ln], dtype=float)
    for idx, line in enumerate(linest):
        line = line.rstrip('\n')
        line = line.rstrip('\t')
        resultsline = line.split('\t')
        del resultsline[0]
        for r in resultsline:
            rt, rv = r.split(' ')
            matrix[tagdic[taggershort + '__' + rt]][idx] = rv

    return matrix, tagaccu, tagset, tagger_tags

def matrixworks(csv, tag_accu, tags, results, tagger_answers, probs, words=None):
    high_pos_t = 0
    high_pos_f = 0
    high_npos_t = 0
    high_npos_f = 0

    jury_pos_t = 0
    jury_pos_f = 0
    jury_npos_t = 0
    jury_npos_f = 0

    xjury_pos_t = 0
    xjury_pos_f = 0
    xjury_npos_t = 0
    xjury_npos_f = 0

    cplx_pos_t = 0
    cplx_pos_f = 0
    cplx_npos_t = 0
    cplx_npos_f = 0

    high_ans = ([])
    jury_ans = ([])
    xjury_ans = ([])
    t_probs = ([])
    f_probs = ([])

    high = True
    jury = True
    xjury = False

    if words is not None:
        words = list(word.rstrip('\n') for word in words if word not in ['\n', '', '\0'])

    with open(csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # All lines including the blank ones

        # print(str(len(lines)))
        # print(str(len(tags)))
        # print(str(len(results)))

        tagset = {}
        normdict = {}
        normdicf = {}
        taggersall = ([])

        for i, tag in enumerate(lines[0].rstrip('\n').split('\t')):
            if tag != 'result':
                tagset[i] = tag.split('__')[1]
                normdict[i] = 0
                normdicf[i] = 0
                tgr = tag.split('__')[0]
                taggersall.append(tgr)

        taggerset = list(set(taggersall))

        del lines[0]

        for idx, line in enumerate(lines):
            values = line.rstrip('\n').split('\t')
            realtag = results[idx].rstrip('\n')
            newtag = tags[idx].rstrip('\n')
            # values[0] = 0.000

            if newtag.split(':')[0] == realtag.split(':')[0]:
                t_probs.append(str(probs[idx]))
                cplx_pos_t += 1
            else:
                cplx_pos_f += 1
                f_probs.append(str(idx + 1) + "|" + str(newtag) + "|" + str(probs[idx]))
            if newtag == realtag or (newtag =="PUNCT" and realtag == "SENT") or (realtag == "PUNCT" and newtag == "SENT"):
                cplx_npos_t += 1

            else:
                cplx_npos_f += 1

            # if realtag != 'SENT' and realtag != 'PUNCT':

            for i, val in enumerate(values):
                values[i] = float(val)

            if high:
                index = values.index(max(values))
                high_guess = tagset[index]
                high_ans.append(high_guess)

                if high_guess.split(':')[0] == realtag.split(':')[0]:
                    high_pos_t += 1
                else:
                    high_pos_f += 1
                if high_guess == realtag or (newtag =="PUNCT" and realtag == "SENT") or (realtag == "PUNCT" and newtag == "SENT"):
                    high_npos_t += 1
                else:
                    high_npos_f += 1

            if jury:
                for i, v in enumerate(values):
                    thistag = tagset[i]
                    for j in range(1, len(tagset)):
                        if tagset[j] == thistag:
                            values[i] += values[j]

                jury_guess = tagset[values.index(max(values))]
                jury_ans.append(jury_guess)

                if jury_guess.split(':')[0] == realtag.split(':')[0]:
                    jury_pos_t += 1
                else:
                    jury_pos_f += 1
                if jury_guess == realtag or (newtag =="PUNCT" and realtag == "SENT") or (realtag == "PUNCT" and newtag == "SENT"):
                    jury_npos_t += 1
                else:
                    jury_npos_f += 1

            if xjury:

                for i, v in enumerate(values):
                    thistag = tagset[i]
                    for j in range(1, len(tagset)):
                        if tagset[j] == thistag:
                            if taggersall[i] + "_" + thistag in tag_accu.keys():
                                accu = tag_accu[taggersall[i] + "_" + thistag]
                            else:
                                accu = 0.5
                            values[i] += values[j] * accu
                xjury_guess = tagset[values.index(max(values))]
                xjury_ans.append(xjury_guess)

                if xjury_guess.split(':')[0] == realtag.split(':')[0]:
                    xjury_pos_t += 1
                else:
                    xjury_pos_f += 1
                if xjury_guess == realtag or (newtag =="PUNCT" and realtag == "SENT") or (realtag == "PUNCT" and newtag == "SENT"):
                    xjury_npos_t += 1
                else:
                    xjury_npos_f += 1

    # print("\t".join(t_probs))
    # print("\t".join(f_probs))
    tagger_rates = {}
    for ta in tagger_answers:

        tagger_rates["pos_t"] = 0
        tagger_rates["pos_f"] = 0
        tagger_rates["npos_t"] = 0
        tagger_rates["npos_f"] = 0

        for i, t in enumerate(tagger_answers[ta]):
            r = results[i].rstrip('\n')
            if t.split(':')[0] == r.split(':')[0]:
                tagger_rates["pos_t"] += 1
            else:
                tagger_rates["pos_f"] += 1
            if t == r or (t =="PUNCT" and r == "SENT") or (r == "PUNCT" and t == "SENT"):
                tagger_rates["npos_t"] += 1
            else:
                tagger_rates["npos_f"] += 1

        rate_pos = 100 / (tagger_rates["pos_t"] + tagger_rates["pos_f"]) * tagger_rates["pos_t"]
        rate_npos = 100 / (tagger_rates["npos_t"] + tagger_rates["npos_f"]) * tagger_rates["npos_t"]
        print(ta + '\t' + str(rate_pos) + '\t' + str(rate_npos))

    if high:
        high_rate_pos = 100 / (high_pos_t + high_pos_f) * high_pos_t
        high_rate_npos = 100 / (high_npos_t + high_npos_f) * high_npos_t
        print('high\t' + str(high_rate_pos) + '\t' + str(high_rate_npos))

    if jury:
        jury_rate_pos = 100 / (jury_pos_t + jury_pos_f) * jury_pos_t
        jury_rate_npos = 100 / (jury_npos_t + jury_npos_f) * jury_npos_t
        print('jury\t' + str(jury_rate_pos) + '\t' + str(jury_rate_npos))

    if xjury:
        xjury_rate_pos = 100 / (xjury_pos_t + xjury_pos_f) * xjury_pos_t
        xjury_rate_npos = 100 / (xjury_npos_t + xjury_npos_f) * xjury_npos_t
        print('xjury\t' + str(xjury_rate_pos) + '\t' + str(xjury_rate_npos))

    cplx_rate_pos = 100 / (cplx_pos_t + cplx_pos_f) * cplx_pos_t
    cplx_rate_npos = 100 / (cplx_npos_t + cplx_npos_f) * cplx_npos_t
    print('cplx\t' + str(cplx_rate_pos) + '\t' + str(cplx_rate_npos))

    with open(csv[:len(csv) - 4] + '_poredjenje.csv', 'w', encoding='utf-8') as f:
        for t in tagger_answers.keys():
            f.write(t + "\t")
        if high:
            f.write("high\t")
        if jury:
            f.write("jury\t")
        if xjury:
            f.write("xjury\t")
        f.write("complex\t")
        f.write("targetPOS")
        if words is not None:
            f.write("\ttargetWord")
        f.write("\n")

        for i, res in enumerate(results):
            for ta in tagger_answers:
                f.write(tagger_answers[ta][i] + "\t")
            if high:
                f.write(high_ans[i] + "\t")
            if jury:
                f.write(jury_ans[i] + "\t")
            if xjury:
                f.write(xjury_ans[i] + "\t")
            f.write(tags[i] + "\t")
            f.write(results[i])
            if words is not None:
                f.write("\t"+words[i])
            f.write('\n')

