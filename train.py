import os
import math
import torch
import yaml
import argparse
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utility import Data
from ACTR import ACTR


def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="1", type=int, help="whith gpu to use")
    parser.add_argument("-l", "--seq_len", default=8, type=int, help="sequence length for each sequential sample")
    parser.add_argument("-d", "--dataset", default="ifashion", type=str, help="which dataset to evaluate")
    parser.add_argument("-r", "--runs", default=1, type=int, help="how many times to conduct the experiments")
    args = parser.parse_args()
    
    return args


def init_best_metrics(conf):
    best_metrics = {}
    best_perform = {}
    for key in ["val", "test"]:
        best_metrics[key] = {}
        best_perform[key] = {}
        for metrix in ["recall", "mrr", "ndcg"]:
            best_metrics[key][metrix] = {}
            for topk in conf["topk"]:
                best_metrics[key][metrix][topk] = 0

    return best_metrics, best_perform


def mk_save_dir(types, root_path, settings):
    output_dir = "./%s/%s"%(types, root_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = output_dir + "%s"%("__".join([str(i) for i in settings]))
    
    return output_file


def write_log(log, key, topk, step, performance):
    for metrix in ["recall", "mrr", "ndcg"]:
        log.add_scalar("%s_%s/%s" %(metrix, topk, key), performance[metrix][topk], step)

    
def train(conf):   
    print("load dataset ...")
    dataset = Data(conf)
    conf["device"] = torch.device("cuda:%d"%conf["gpu"] if torch.cuda.is_available() else "cpu")
    conf["usernum"] = dataset.usernum
    conf["itemnum"] = dataset.itemnum
    conf["relnum"] = len(conf["Relationships"])
    conf["ttl_meta_value"] = len(dataset.id_attr_map)
    root_path = "%s/seq_len_%d/ACTR/" %(conf['dataset'].upper(), conf["seq_len"])
    settings = ["run_%d"%conf["run_cnt"]]
    model = ACTR(conf, dataset.iid_grouped_meta)
    model.to(device=conf["device"])
    performance_file = mk_save_dir("performance", root_path, settings)
    result_file = mk_save_dir("results", root_path, settings)
    log_file = mk_save_dir("logs", root_path, settings)
    log = SummaryWriter(log_file)    
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["weight_decay"])
    best_metrics, best_perform = init_best_metrics(conf)
    best_epoch = 0
    for epoch in range(conf["num_epoches"]):
        loss_print, rel_loss_print, seq_loss_print, item_loss_print = 0, 0, 0, 0
        for batch_cnt, batch in enumerate(dataset.train_loader):
            model.train(True)
            optimizer.zero_grad()
            batch = [x.to(conf["device"]) for x in batch]
            loss, relation_loss, seq_loss, item_loss = model(batch)
            loss_scalar = loss.detach()
            rel_loss_scalar = relation_loss.detach()
            seq_loss_scalar = seq_loss.detach()
            item_loss_scalar = item_loss.detach()
            item_loss_print += item_loss_scalar
            loss_print += loss_scalar
            rel_loss_print += rel_loss_scalar
            seq_loss_print += seq_loss_scalar
            loss.backward()
            optimizer.step()
            log.add_scalar("Loss/Loss", loss_scalar, batch_cnt+epoch*len(dataset.train_set)/conf['batch_size'])
            log.add_scalar("Loss/Relation Loss", rel_loss_scalar, batch_cnt+epoch*len(dataset.train_set)/conf['batch_size'])
            log.add_scalar("Loss/Seq Loss", seq_loss_scalar, batch_cnt+epoch*len(dataset.train_set)/conf['batch_size'])
            log.add_scalar("Loss/Item Loss", item_loss_scalar, batch_cnt+epoch*len(dataset.train_set)/conf['batch_size'])
        
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_str = '%s  epoch %d, %s/len_%d/%s/%s, loss %.4f, rel_loss %.4f, seq_loss %.4f, item_loss %.4f' %(curr_time, epoch, conf["dataset"], conf["seq_len"], "ACTR", "_".join([str(i) for i in settings]), loss_print/(batch_cnt+1), rel_loss_print/(batch_cnt+1), seq_loss_print/(batch_cnt+1), item_loss_print/(batch_cnt+1))
        print(log_str)
        if (epoch + 1) % conf["test_interval"] == 0:
            model.eval()
            best_metrics, best_perform, best_epoch = evaluate(model, dataset, conf, log, performance_file, result_file, epoch, batch_cnt, best_metrics, best_perform, best_epoch)
            
            
def evaluate(model, dataset, conf, log, performance_file, result_file, epoch, batch_cnt, best_metrics, best_perform, best_epoch): 
    metrics = {}
    item_id_rep = model.itemEmbedding.weight
    item_id_rep_r = model.itemEmbedding_r.weight  
    step = batch_cnt + epoch*dataset.train_len/conf["batch_size"]
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
    loaderset = {"val":dataset.val_loader, "test":dataset.test_loader}
    for key in ["val", "test"]:
        metrics[key] = rank(model, loaderset[key], item_id_rep, item_id_rep_r, conf) 
        for topk in conf["topk"]:
            print("%s, %s  res: rec_%d: %f, mrr_%d: %f, ndcg_%d: %f" %(curr_time, key, topk, metrics[key]["recall"][topk], topk, metrics[key]["mrr"][topk], topk, metrics[key]["ndcg"][topk]))
            write_log(log, key, topk, step, metrics[key])

    topk_ = conf["topk"][0]
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > best_metrics["val"]["ndcg"][topk_] and metrics["val"]["mrr"][topk_] > best_metrics["val"]["mrr"][topk_]:
        best_epoch = epoch
        output_f = open(performance_file, "a")
        for topk in conf['topk']:
            for key in best_metrics:
                for metric in best_metrics[key]:                    
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]
            
                best_perform[key][topk] = "%s, Best in epoch %d, TOP %d: REC_%s=%.5f, MRR_%s=%.5f, NDCG_%s=%.5f"%(curr_time, best_epoch, topk, key, best_metrics[key]["recall"][topk], key, best_metrics[key]["mrr"][topk], key, best_metrics[key]["ndcg"][topk])
                output_f.write(best_perform[key][topk] + "\n")
        output_f.write("\n")
        output_f.close()  
    print(best_perform["val"][topk_])
    print(best_perform["test"][topk_])

    return best_metrics, best_perform, best_epoch


def rank(model, test_loader, item_id_rep, item_id_rep_r, conf):
    device = conf["device"]
    trg_item_pred = {}
    for batch_cnt, batch in enumerate(test_loader):
        [u, i, ir, j, ks] = [x.to(device) for x in batch]
        u_rep = model.userEmbedding(u) #[test_batch_size, hidden_size]
        i_rep = item_id_rep[i].to(device)
        j_rep = item_id_rep[j].to(device)
        i_meta_rep = model.MetaEmbedding(model.item_meta[i])
        j_meta_rep = model.MetaEmbedding(model.item_meta[j])
        i_rep_rel = item_id_rep_r[i].to(device)
       
        r_rep = model.relationEmbedding.weight.unsqueeze(0).expand(u_rep.size()[0],-1,-1) # test_bs, 1, hidden_size
        relation_user_embed = torch.cat([r_rep, u_rep.unsqueeze(1)], 1) # test_bs, relnum+2, hidden_size
        relation_weight = model.predict_relation(u_rep, i_rep_rel, r_rep)

        pos_score_0 = model.predict_score(u_rep, i_rep, j_rep, relation_weight, relation_user_embed, i_meta_rep, j_meta_rep)
        pos_score =(model.itemBiases(j).squeeze(-1) + pos_score_0).detach()
        pos_score = pos_score.unsqueeze(1) # [batch_size, 1]
        neg_set_num = ks.size()[1]
        for neg_set_cnt in range(neg_set_num):
            if neg_set_cnt not in trg_item_pred:
                trg_item_pred[neg_set_cnt] = []
            k = ks[:, neg_set_cnt, :]  
            neg_scores = []
            for k_cnt in range(k.size()[-1]):
                kk = k[:,k_cnt]
                k_rep = item_id_rep[kk].to(device)   
                k_meta_rep = model.MetaEmbedding(model.item_meta[kk])
                neg_scores_0 = model.predict_score(u_rep, i_rep, k_rep, relation_weight, relation_user_embed, i_meta_rep, k_meta_rep)
                neg_scores.append((model.itemBiases(kk).squeeze(-1) + neg_scores_0).detach())
            
            neg_scores = torch.stack(neg_scores, dim=1)
            score = torch.cat([pos_score, neg_scores], dim=-1)
            top_score, tops = torch.topk(score, k=max(conf["topk"]), dim=-1)
            trg_item_pred[neg_set_cnt].append(tops.cpu().numpy())
            
    pred = {}
    for neg_set_cnt in range(neg_set_num):
        pred[neg_set_cnt] = np.concatenate(trg_item_pred[neg_set_cnt], axis=0)
        
    grd = [0]*np.shape(pred[0])[0]
    grd_cnt = [1]*np.shape(pred[0])[0]

    REC, MRR, NDCG = get_metrics(grd, grd_cnt, pred, conf["neg_set_num"], conf["topk"])  
    metrics = {}
    metrics["recall"] = REC
    metrics["mrr"] = MRR
    metrics["ndcg"] = NDCG
    
    return metrics


def get_metrics(grd, grd_cnt, pred, neg_set_num, topks):
    REC, MRR, NDCG = {}, {}, {}
    for topk in topks:
        REC[topk] = []
        MRR[topk] = []
        NDCG[topk] = []
        for neg_cnt in range(neg_set_num):
            rec_, mrr_, ndcg_ = [], [], []
            for each_grd, each_grd_cnt, each_pred in zip(grd, grd_cnt, pred[neg_cnt]):
                ndcg_.append(getNDCG(each_pred[:topk], [each_grd][:each_grd_cnt]))
                hit, mrr = getHIT_MRR(each_pred[:topk], [each_grd][:each_grd_cnt])
                rec_.append(hit)
                mrr_.append(mrr)
            mrr_ = np.mean(mrr_)
            rec_ = np.mean(rec_)
            ndcg_ = np.mean(ndcg_)
            REC[topk].append(rec_)
            MRR[topk].append(mrr_)
            NDCG[topk].append(ndcg_)
        REC[topk] = np.mean(REC[topk])
        MRR[topk] = np.mean(MRR[topk])
        NDCG[topk] = np.mean(NDCG[topk])

    return REC, MRR, NDCG


def getHIT_MRR(pred, target_items):
    hit= 0.
    mrr = 0.
    p_1 = []
    for p in range(len(pred)):
        pre = pred[p]
        if pre in target_items:
            hit += 1
            if pre not in p_1:
                p_1.append(pre)
                mrr = 1./(p+1)
                
    return hit, mrr


def getNDCG(rank_list, target_items):
    dcg = 0
    idcg = IDCG(len(target_items))
    for i in range(len(rank_list)):
        item_id = rank_list[i]
        if item_id not in target_items:
            continue
        rank = i + 1
        dcg += 1./math.log(rank+1, 2)
        
    return dcg/idcg


def IDCG(n):
    idcg = 0.
    for i in range(n):
        idcg += 1./math.log(i+2, 2)
        
    return idcg
    

def main():
    conf = yaml.safe_load(open("./config.yaml"))
    paras = get_cmd().__dict__    
    for k, v in paras.items():
        conf[k] = v
    dataset = conf["dataset"]
    for k, v in conf[dataset].items():
        conf[k] = v
    for run_cnt in range(conf["runs"]):
        conf["run_cnt"] = run_cnt
        train(conf)


if __name__ == "__main__":
    main()
