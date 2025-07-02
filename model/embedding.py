import math
import torch.nn.functional as F
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        pos_encoding = torch.zeros(max_len, self.emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.emb_dim, 2).float() * -(math.log(10000.0) / self.emb_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        self.dropout = nn.Dropout(0.1)

    def forward(self, out):
        out = out + self.pos_encoding[:, :out.size(1)].detach()
        out = self.dropout(out)
        return out


class MyEmbedding(nn.Module):
    def __init__(self, config, args):
        super(MyEmbedding, self).__init__()
        self.config = config
        self.args = args

        self.num_locations = config.Dataset.num_locations
        self.num_prototypes = args.num_prototypes
        self.base_dim = args.base_dim
        self.prefer_dim = args.prefer_dim
        self.num_users = config.Dataset.num_users
        self.num_timeslots = args.num_timeslots

        self.user_embedding = nn.Embedding(self.num_users, self.base_dim)
        self.prefer_embedding = nn.Embedding(self.num_users, self.prefer_dim)
        self.location_embedding = nn.Embedding(self.num_locations, self.base_dim)
        self.timeslot_embedding = nn.Embedding(self.num_timeslots, self.base_dim)

        if self.num_prototypes > 0:
            self.prototypes_embedding = nn.Embedding(self.num_prototypes, self.base_dim)
            self.q_proj_loc = nn.Linear(self.base_dim, self.base_dim)
            self.k_proj_loc = nn.Linear(self.base_dim, self.base_dim)
            input_dim = self.base_dim * 1
            hidden_dim = input_dim
            self.ffn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.base_dim),
            )

            self.norm = nn.LayerNorm(self.base_dim)

    def forward(self, batch_data, aux_mat):
        user_ids = batch_data['user']
        timeslot_ids = batch_data['timeslot']

        loc_emb_all = self.location_embedding.weight
        user_emb = self.user_embedding(user_ids)
        time_emb = self.timeslot_embedding(timeslot_ids)
        prefer_emb = self.prefer_embedding(user_ids)
        loc_proto_emb_enhanced = None
        loc_proto_emb = None

        user_emb = user_emb.unsqueeze(1).expand_as(time_emb)

        if self.num_prototypes > 0:
            proto_emb = self.prototypes_embedding.weight

            q = self.q_proj_loc(loc_emb_all)  # Shape: (num_locations, base_dim)

            proto_emb_k_loc = proto_emb  # Shape: (num_prototypes, base_dim)
            proto_emb_k_loc = self.k_proj_loc(proto_emb)  # Shape: (num_prototypes, base_dim)
            weight_scores_loc = torch.matmul(q, proto_emb_k_loc.T)  # Scaled dot-product, Shape: (num_locations, num_prototypes)

            weights_loc = F.softmax(weight_scores_loc, dim=-1)  # Normalize scores, Shape: (num_locations, num_prototypes)
            loc_proto_emb = torch.matmul(weights_loc, proto_emb)  # Shape: (num_locations, base_dim)

            fusion_input = loc_proto_emb + loc_emb_all
            loc_proto_emb_enhanced = self.ffn(fusion_input)
         
            loc_proto_emb_enhanced = self.norm(loc_proto_emb_enhanced+loc_proto_emb)


        return loc_emb_all, time_emb, user_emb, loc_proto_emb_enhanced, prefer_emb, loc_proto_emb
