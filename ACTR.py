import torch
import torch.nn as nn
import torch.nn.functional as F

def bpr_loss(pos_score, neg_score):
    loss = - F.logsigmoid(pos_score - neg_score)
    loss = torch.mean(loss)
    
    return loss


class ACTR(nn.Module):
    def __init__(self, conf, item_meta):
        super(ACTR, self).__init__()
        self.conf = conf
        self.usernum = self.conf['usernum']
        self.itemnum = self.conf['itemnum']
        self.relnum = self.conf['relnum']
        self.d_emb = self.conf['embedding_dim']
        self.gamma = self.conf['gamma']
        self.alpha = self.conf["alpha"]
        self.beta = self.conf["beta"]
        self.item_meta = torch.LongTensor(item_meta).to(self.conf['device'])
        self.itemB = torch.zeros([self.itemnum, 1])
        self.relB = torch.zeros([self.relnum, 1])
        self.userEmb = F.normalize(torch.normal(mean=torch.zeros(self.usernum, self.d_emb), std=1/(self.d_emb)**0.5), p=2, dim=-1)
        self.relationEmb = F.normalize(torch.normal(mean=torch.zeros(self.relnum, self.d_emb), std=1/(self.d_emb)**0.5), p=2, dim=-1)
        self.itemEmb = F.normalize(torch.normal(mean=torch.zeros(self.itemnum, self.d_emb), std=1/(self.d_emb)**0.5), p=2, dim=-1)
        self.itemEmb_r = F.normalize(torch.normal(mean=torch.zeros(self.itemnum, self.d_emb), std=1/(self.d_emb)**0.5), p=2, dim=-1)
        self.itemBiases = nn.Embedding.from_pretrained(self.itemB, freeze=False)
        self.relationBiases = nn.Embedding.from_pretrained(self.relB, freeze=False)
        self.userEmbedding = nn.Embedding.from_pretrained(self.userEmb, freeze=False)
        self.relationEmbedding = nn.Embedding.from_pretrained(self.relationEmb, freeze=False) 
        self.itemEmbedding = nn.Embedding.from_pretrained(self.itemEmb, freeze=False)
        self.itemEmbedding_r = nn.Embedding.from_pretrained(self.itemEmb_r, freeze=False)
        self.rel_map = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).to(self.conf['device'])
        self.MetaEmbedding = nn.Embedding(self.conf['ttl_meta_value'] + 1, self.d_emb, padding_idx=0)
        self.uc_att_w = nn.Linear(self.d_emb, self.d_emb)
        self.uc_att_v = nn.Linear(self.d_emb * 2, 1)

        
    def get_attention(self, u_rep, i_rep_plus):
        wu = self.uc_att_w(u_rep) # bs, d_emb
        wic = self.uc_att_w(i_rep_plus) # bs, meta_num+1, d_emb
        wu_ic = torch.cat([wu, wic], dim=-1)
        v_u_ic = self.uc_att_v(wu_ic).squeeze(-1) 
        uc_coef = F.softmax(v_u_ic, dim=-1)
        
        return uc_coef # bs, relnum, meta_num+1

    
    def predict_score(self, u_rep, i_rep, j_rep, relation, relation_emb, i_rep_meta, j_rep_meta):
        if len(i_rep.size()) == 2:
            i_rep = i_rep.unsqueeze(1).unsqueeze(1).expand(-1, self.relnum+1, -1, -1)
            j_rep = j_rep.unsqueeze(1).unsqueeze(1).expand(-1, self.relnum+1, -1, -1)
            i_rep_meta = i_rep_meta.unsqueeze(1).expand(-1, self.relnum+1, -1, -1)
            j_rep_meta = j_rep_meta.unsqueeze(1).expand(-1, self.relnum+1, -1, -1)
            relation_emb = relation_emb.unsqueeze(-2).expand(-1, -1, self.conf['meta_group_num']+1, -1)
            relation = torch.cat([relation * self.gamma, (1 - self.gamma) * torch.ones([relation.size()[0], 1]).to(self.conf["device"])], 1)
        elif len(i_rep.size()) == 3:
            i_rep = i_rep.unsqueeze(2).unsqueeze(2).expand(-1, -1, self.relnum+1, -1, -1) # bs, neg_num, rel+1, meta_group+1, d_emb
            j_rep = j_rep.unsqueeze(2).unsqueeze(2).expand(-1, -1, self.relnum+1, -1, -1)
            i_rep_meta = i_rep_meta.unsqueeze(2).expand(-1, -1, self.relnum+1, -1, -1)
            j_rep_meta = j_rep_meta.unsqueeze(2).expand(-1, -1, self.relnum+1, -1, -1)
            relation_emb = relation_emb.unsqueeze(-2).expand(-1, -1, -1, self.conf['meta_group_num']+1, -1)
            relation = torch.cat([relation * self.gamma, (1 - self.gamma) * torch.ones([relation.size()[0], relation.size()[1], 1]).to(self.conf["device"])], 2)
        i_rep_plus = torch.cat([i_rep, i_rep_meta], dim=-2)
        j_rep_plus = torch.cat([j_rep, j_rep_meta], dim=-2)   
        
        if len(i_rep.size()) == 4:
            u_rep = u_rep.unsqueeze(1).unsqueeze(1).expand(-1, self.relnum+1, self.conf['meta_group_num']+1, -1)
        elif len(i_rep.size()) == 5:
            u_rep = u_rep.unsqueeze(2).unsqueeze(2).expand(-1, -1, self.relnum+1, self.conf['meta_group_num']+1, -1)
        uc_att_coef = self.get_attention(u_rep, i_rep_plus)
        
        return torch.sum(relation*(-torch.sum(uc_att_coef * torch.sum(torch.pow((i_rep_plus + relation_emb - j_rep_plus), 2), -1), -1)), -1)
        
        
    def predict_relation(self, u_rep, i_rep, batch_r_embed):
        bs = u_rep.size()[0]
        u_rep = u_rep.unsqueeze(1).expand(-1, self.relnum, -1) # bs, relnum, emb_size
        i_rep = i_rep.unsqueeze(1).expand(-1, self.relnum, -1)
        batch_r_bias = self.relationBiases.weight.permute(1, 0).expand(bs, -1) # bs,relnum
        pred = batch_r_bias - torch.sum(torch.pow((u_rep + i_rep - batch_r_embed), 2), -1) # bs, relnum+1
        pred_soft = F.softmax(pred, dim=-1) # bs, relnum+1

        return pred_soft    

    
    def forward(self, batch):
        u_id, anchor_i_id, pos_r_id, pos_i_id, neg_r_id, neg_i_id, neg_ri_id = batch
        u_rep = self.userEmbedding(u_id) # bs, emb_size
        anchor_i_rep = self.itemEmbedding(anchor_i_id)
        pos_i_rep = self.itemEmbedding(pos_i_id)
        neg_i_rep = self.itemEmbedding(neg_i_id)
        neg_ri_rep = self.itemEmbedding(neg_ri_id)
        anchor_i_meta = self.item_meta[anchor_i_id] # bs, num_meta_group (24)
        pos_i_meta = self.item_meta[pos_i_id] # bs, num_meta_group (24) 
        neg_i_meta = self.item_meta[neg_i_id] # bs, num_meta_group (24)
        anchor_i_rep_meta = self.MetaEmbedding(anchor_i_meta) # bs, num_meta_group,  emb_size 
        pos_i_rep_meta = self.MetaEmbedding(pos_i_meta) # bs, num_meta_group,  emb_size  
        neg_i_rep_meta = self.MetaEmbedding(neg_i_meta) # bs, num_meta_group,  emb_size  
        bs = u_id.size()[0]
        batch_r_rep = self.relationEmbedding.weight.unsqueeze(0).expand(bs, -1, -1)
        
        anchor_i_rep_rel = self.itemEmbedding_r(anchor_i_id)
        relation_weight_all = self.predict_relation(u_rep, anchor_i_rep_rel, batch_r_rep) # bs, relnum+1        
        relation_weight = relation_weight_all
        relation_user_embed = torch.cat([batch_r_rep, u_rep.unsqueeze(1)], 1) # bs,relnum+2,emb_size
        
        pos_R = self.itemBiases(pos_i_id).squeeze(-1) + self.predict_score(u_rep, anchor_i_rep, pos_i_rep, relation_weight, relation_user_embed, anchor_i_rep_meta, pos_i_rep_meta)
        neg_R = self.itemBiases(neg_i_id).squeeze(-1) + self.predict_score(u_rep, anchor_i_rep, neg_i_rep, relation_weight, relation_user_embed, anchor_i_rep_meta, neg_i_rep_meta)
        seq_loss = bpr_loss(pos_R, neg_R)
            
        pos_r = self.rel_map[pos_r_id]
        neg_r = self.rel_map[neg_r_id]
        pos_r_score = torch.sum(relation_weight * pos_r, dim=-1)
        neg_r_score = torch.sum(relation_weight * neg_r, dim=-1)
        relation_loss = bpr_loss(pos_r_score, neg_r_score)
 
        relation_embed = torch.cat([self.relationEmbedding.weight], 0) # relnum+1,emb_size
        iir_rep = relation_embed[pos_r_id]
        pos_i_score = self.itemBiases(pos_i_id).squeeze(-1) - torch.sum(torch.pow((anchor_i_rep + iir_rep - pos_i_rep), 2), 1)
        neg_i_score = self.itemBiases(neg_ri_id).squeeze(-1) - torch.sum(torch.pow((anchor_i_rep + iir_rep - neg_ri_rep), 2), 1)
        item_loss = bpr_loss(pos_i_score, neg_i_score)

        loss = seq_loss + self.beta * relation_loss + self.alpha * item_loss

        return loss, relation_loss, seq_loss, item_loss