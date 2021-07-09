#coding: utf-8
import json
import random
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
random.seed(1234)

class TrainData(Dataset):
    def __init__(self, conf, input_seq, r_list, all_itmes, user_items, match_cate_set, item_cate_set):
        self.input_seq = input_seq
        self.conf = conf
        self.r_list = r_list
        self.item_cate_set = item_cate_set
        self.match_cate_set = match_cate_set
        self.user_items = user_items
        self.all_items = all_itmes
        
    def __len__(self):
        return len(self.input_seq)
    
    def get_relation(self, i, j, item_cate_set, match_cate_set):
        # relation: others:0, match:1, sub:2
        i = str(i)
        j = str(j)
        [cate_i_1, cate_i_2] = item_cate_set[i]
        [cate_j_1, cate_j_2] = item_cate_set[j]
        if cate_i_2 == cate_j_2:
            r = 2
        elif (cate_i_1 in match_cate_set and cate_j_1 in match_cate_set[cate_i_1]) \
            or (cate_i_1 in match_cate_set and cate_j_2 in match_cate_set[cate_i_1]) \
            or (cate_i_2 in match_cate_set and cate_j_1 in match_cate_set[cate_i_2]) \
            or (cate_i_2 in match_cate_set and cate_j_2 in match_cate_set[cate_i_2]):
                r = 1
        else:
            r = 0
            
        return r
    
    
    def __getitem__(self, idx):
        seq = self.input_seq[idx]
        u = seq[-1]
        j = seq[-2]
        i = seq[-3]
        r = int(self.r_list[idx])
        
        user_seq = self.user_items[str(u)]
        k = random.sample(self.all_items, 1)[0]
        while k in user_seq:
            k = random.sample(self.all_items, 1)[0]
            
        rk = random.sample(self.all_items, 1)[0]
        rk_rel = self.get_relation(i, rk, self.item_cate_set, self.match_cate_set)
        while rk_rel == r:
            rk = random.sample(self.all_items, 1)[0]
            rk_rel = self.get_relation(i, rk, self.item_cate_set, self.match_cate_set)
   
        r_neg = random.randint(0, 2)
        while r_neg == r:
            r_neg = random.randint(0, 2)   
        
        return u, i, r, j, int(r_neg), int(k), int(rk)  

  
class TestData(Dataset):
    def __init__(self, conf, input_seqs, neg_seqs, r_list):
        self.conf = conf
        self.input_seq = input_seqs
        self.neg_seqs = neg_seqs # list of neg samples for each item
        self.r_list = r_list

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, idx):
        each = self.input_seq[idx]
        u = each[-1]
        j = each[-2]
        i = each[-3]
        ks = [self.neg_seqs[cnt][idx] for cnt in self.neg_seqs.keys()]
        r = int(self.r_list[idx])
        ks = torch.LongTensor(ks)

        return int(u), i, r, j, ks


class Data():
    def __init__(self, conf):
        self.conf = conf
        self.target_path = self.conf['target_path']
        self.match_cate_set = load_cate_match_pair(self.conf['datapath'])
        all_data = load_cache_data(self.target_path, self.conf)
        self.train_seqs, self.val_seqs, self.val_negs, self.test_seqs, self.test_negs, self.user_id_map, self.id_user_map, self.item_id_map, self.id_item_map, self.user_items = all_data
        self.item_ids = list(self.id_item_map.keys())
        self.usernum = len(self.user_items)
        self.itemnum = len(self.item_id_map)
        self.id_attr_map = json.load(open(self.target_path + "id_meta_map.json"))
        print("%s: total item number: %d" %(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.itemnum))
        print("train samples: ", len(self.train_seqs))
        print("test samples: ", len(self.test_seqs))
        print("val samples: ", len(self.val_seqs))
        self.iid_grouped_meta = load_item_grouped_meta(self.target_path, self.conf, self.item_id_map)
        self.cate_id_map, self.id_cate_map, self.item_two_cate_ids, self.item_two_cates, self.cate_item_set = get_cates_cache(self.target_path, self.conf)
        self.match_cate_id_set = map_match_cate(self.match_cate_set, self.cate_id_map)
        self.train_seq_rel, self.test_seq_rel, self.val_seq_rel = load_rel_and_neg4train(self.target_path, self.conf)

        self.train_set = TrainData(self.conf, self.train_seqs, self.train_seq_rel, self.item_ids, self.user_items, self.match_cate_id_set, self.item_two_cate_ids)
        self.train_loader = DataLoader(self.train_set, batch_size=self.conf["batch_size"], shuffle=True, num_workers=self.conf['num_workers'])
        self.test_set = TestData(self.conf, self.test_seqs, self.test_negs, self.test_seq_rel)
        self.test_loader = DataLoader(self.test_set, batch_size=self.conf["test_batch_size"], shuffle=False, num_workers=self.conf['num_workers'])
        self.val_set = TestData(self.conf, self.val_seqs, self.val_negs, self.val_seq_rel)
        self.val_loader = DataLoader(self.val_set, batch_size=self.conf["test_batch_size"], shuffle=False, num_workers=self.conf['num_workers'])
        self.train_len = len(self.train_seqs)


def load_rel_and_neg4train(target_path, conf):
    train_rel = json.load(open(target_path + "train_rel_%d.json"%( conf['seq_len'])))
    test_rel = json.load(open(target_path + "test_rel_%d.json"%(conf['seq_len'])))
    val_rel = json.load(open(target_path + "val_rel_%d.json"%(conf['seq_len'])))
    
    return train_rel, test_rel, val_rel


def load_item_grouped_meta(target_path, conf, item_id_map):
    item_grouped_meta = json.load(open(target_path + "item_grouped_meta_list.json"))
    iid_grouped_meta = {}
    for item in item_id_map:
        iid = item_id_map[item]
        if item in item_grouped_meta:
            grouped_meta = item_grouped_meta[item]
        else:
            grouped_meta = [0] * 24
        iid_grouped_meta[iid] = grouped_meta
    iid_group_meta_list = []
    for iid in sorted(iid_grouped_meta.keys()):
        iid_group_meta_list.append(iid_grouped_meta[iid])
        
    return iid_group_meta_list
                        
        
def load_cate_match_pair(target_path):
    match_pairs = json.load(open(target_path + "top_match_cates_550_0095.json"))
    cate_match_pair =  {}
    for pair in match_pairs:
        cate_1 = pair[0]
        cate_2 = pair[1]
        if cate_1 not in cate_match_pair:
            cate_match_pair[cate_1] = []
        if cate_2 not in cate_match_pair:
            cate_match_pair[cate_2] = []
        cate_match_pair[cate_1].append(cate_2)
        cate_match_pair[cate_2].append(cate_1)
        
    return cate_match_pair
        
    
def map_match_cate(match_cate_set, cate_id_map):
    match_cate_id_set = {}
    for cate1 in match_cate_set:
        cid1 = cate_id_map[cate1]
        match_cate_id_set[cid1] = [cate_id_map[cate2] for cate2 in match_cate_set[cate1]]
        
    return match_cate_id_set


def get_cates_cache(target_path, conf):
    [cate_id_map, id_cate_map] = json.load(open(target_path + "cate_id_map_%d.json"%(conf['seq_len']), "r"))  
    [item_two_cates, item_two_cate_ids] = json.load(open(target_path + "item_cate_set_%d.json"%(conf['seq_len']), "r")) 
    cate_item_set = json.load(open(target_path + "cate_item_set_%d.json"%(conf['seq_len']), "r"))
    
    return cate_id_map, id_cate_map, item_two_cate_ids, item_two_cates, cate_item_set


def load_cache_data(target_path, conf):
    val_negs = {}
    test_negs = {}
    train_seqs = json.load(open(target_path + "train_seqs_%d.json"%conf['seq_len']))
    if conf["dataset"] == "ifashion":
        val_seqs, val_negs[0] = json.load(open(target_path + "val_seqs_%d.json"%conf['seq_len']))
        test_seqs, test_negs[0] = json.load(open(target_path + "test_seqs_%d.json"%conf['seq_len']))
    elif conf["dataset"] == "amazon":
        val_seqs= json.load(open(target_path + "val_seqs_%d.json"%conf['seq_len']))
        test_seqs = json.load(open(target_path + "test_seqs_%d.json"%conf['seq_len']))
        val_negs[0] = json.load(open(target_path + "val_negs_%d.json"%conf['seq_len']))["0"]
        test_negs[0] = json.load(open(target_path + "test_negs_%d.json"%conf['seq_len']))["0"]
    user_id_map = json.load(open(target_path + "user_id_map_%d.json"%conf['seq_len']))
    id_user_map = json.load(open(target_path + "id_user_map_%d.json"%conf['seq_len']))
    item_id_map = json.load(open(target_path + "item_id_map_%d.json"%conf['seq_len']))
    id_item_map = json.load(open(target_path + "id_item_map_%d.json"%conf['seq_len']))
    user_item_set = json.load(open(target_path + "user_item_set_%d.json"%conf['seq_len']))  
    
    return train_seqs, val_seqs, val_negs, test_seqs, test_negs, user_id_map, id_user_map, item_id_map, id_item_map, user_item_set
