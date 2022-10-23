from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from transformers import BartTokenizer
import random
import pickle
from clustering import get_labels, oracle_clustering


PROMPTS = [
    "First, ",
    "Second, ",
    "Third, ",
    "Fourth, ",
    "Fifth, ",
    "Sixth, ",
    "Seventh, ",
    "Eighth, ",
    "Ninth, ",
    "Tenth, ",
    "Eleventh, ",
    "Twelfth, ",
    "Thirteenth, ",
    "Fourteenth, ",
    "Fifteenth, ",
    "Sixteenth, ",
    "Seventeenth, ",
    "Eighteenth, ",
    "Nineteenth, ",
    "Twentieth, ",
]

def compute_mask(lengths):
    lengths = lengths.cpu()
    max_len = int(torch.max(lengths).item())
    range_row = torch.arange(0, max_len).long()[None, :].expand(lengths.size(0), max_len)
    mask = lengths[:, None].expand_as(range_row).long()
    mask = range_row < mask
    mask = mask.float()
    return mask

def bert_pad(X, pad_token_id=0, max_len=-1):
    if max_len < 0:
        max_len = max(len(x) for x in X)
    result = []
    for x in X:
        if len(x) < max_len:
            x.extend([pad_token_id] * (max_len - len(x)))
        result.append(x)
    return torch.LongTensor(result)

def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data":
            batch[n] = batch[n].to(gpuid)


class MyDataset(Dataset):
    def __init__(self, fdir, model_type, maxlen=512, is_test=False, total_len=512, tgt_max_len=256, num_clusters=5, is_base=False, is_json=False, is_random=False, cluster_type=None, prompting=False):
        """ data format: article, abstract, label (optional) """
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            self.num = len(os.listdir(fdir))
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            self.num = len(self.files)
        self.tok = BartTokenizer.from_pretrained(model_type, verbose=False)
        self.maxlen = maxlen
        self.is_test = is_test
        self.total_len = total_len
        self.tgt_max_len = tgt_max_len
        self.num_clusters = num_clusters
        self.is_base = is_base
        self.is_json = is_json
        self.is_random = is_random
        self.cluster_type = cluster_type
        self.labels = [_ for _ in range(self.num_clusters)]
        self.prompting = prompting
        if self.prompting:
            self.prompts = PROMPTS

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json"%idx), "rb") as f:
                if self.is_json:
                    data = json.load(f)
                else:
                    data = pickle.load(f)
        else:
            with open(self.files[idx], "rb") as f:
                if self.is_json:
                    data = json.load(f)
                else:
                    data = pickle.load(f)

        if self.is_base:
            num = len(data["article"]) // self.num_clusters
            label = []
            for i in range(self.num_clusters):
                label.extend([i] * num)
            while len(label) < len(data["article"]):
                label.append(self.num_clusters - 1)
        elif self.is_random:
            label = []
            if len(data["article"]) > 0:
                label = random.choices(self.labels, k = len(data["article"]))
        elif self.cluster_type == "oracle":
            if len(data["article"]) < self.num_clusters:
                label = np.array([_ for _ in range(len(data["article"]))])
            else:
                label = oracle_clustering(data["article"], data["abstract"], data["similarities"], n_clusters=self.num_clusters)
        else:
            if "label" not in data.keys():
                if len(data["article"]) < self.num_clusters:
                    label = np.array([_ for _ in range(len(data["article"]))])
                else:
                    label = get_labels(data["similarities"], n_clusters=self.num_clusters)
            else:
                label = data["label"]

        if self.prompting:
            article = [[self.prompts[i]] for i in range(self.num_clusters)]
        else:
            article = [[] for _ in range(self.num_clusters)]
        
        if self.cluster_type == "multi_doc":
            for i in range(min(len(data["article"]), self.num_clusters)):
                article[i].append(data["article"][i])
        else:
            for (x, y) in zip(data["article"], label):
                article[y].append(x)

        for i in range(self.num_clusters):
            if len(article[i]) == 0:
                article[i].append(".")

        try:
            article = [" ".join(x) for x in article]
        except:
            article = ["." for _ in range(self.num_clusters)]
        
        src = self.tok.batch_encode_plus(article, max_length=self.maxlen, return_tensors="pt", padding="max_length", truncation=True)
        src_input_ids = src["input_ids"]
        abstract = data["abstract"]
        abstract = " ".join(abstract)
        tgt = self.tok.batch_encode_plus([abstract], max_length=self.tgt_max_len, return_tensors="pt", padding="max_length", truncation=True)
        tgt_input_ids = tgt["input_ids"]
        result = {
            "src_input_ids": src_input_ids, 
            "tgt_input_ids": tgt_input_ids,
            }
        if self.is_test:
            result["data"] = data
            result["data"]["pages"] = article
        return result


def collate_mp(batch, pad_token_id, is_test=False):
    def mat_pad(X):
        seq_num = max([x.size(0) for x in X])
        result = torch.ones(len(X), seq_num, X[0].size(1), dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result

    src_input_ids = mat_pad([x["src_input_ids"] for x in batch])
    tgt_input_ids = torch.cat([x["tgt_input_ids"] for x in batch])
    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "tgt_input_ids": tgt_input_ids,
        }
    if is_test:
        result["data"] = data
    return result



