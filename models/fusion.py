import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder


class fusionModel(nn.Module):
    def __init__(self, hyp_params):
        super(fusionModel, self).__init__()
        self.orig_d_p, self.orig_d_v1, self.orig_d_v2 = hyp_params.orig_d_p, hyp_params.orig_d_v1, hyp_params.orig_d_v2
        #40,2,3
        self.d_p, self.d_v1, self.d_v2 = 50, 50, 50 #token length
        #######
        self.v1only = hyp_params.v1only
        self.v2only = hyp_params.v2only
        self.ponly = hyp_params.ponly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_v1 = hyp_params.attn_dropout_v1#default 0.1
        self.attn_dropout_v2 = hyp_params.attn_dropout_v2
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        
        self.dataset = hyp_params.dataset
        
        combined_dim = self.d_p + self.d_v1 + self.d_v2
        #! 这三者都是True
        self.partial_mode = self.ponly + self.v1only + self.v2only
        if self.partial_mode == 1:
            if self.ponly:
                combined_dim = self.d_p
            if self.v2only:
                combined_dim = self.d_v2
            if self.v1only:
                combined_dim = self.d_v1
        elif self.partial_mode == 2:
            if self.ponly and self.v2only:
                combined_dim = 2 * (self.d_p + self.d_v2)
            if self.ponly and self.v1only:
                combined_dim = 2 * (self.d_p + self.d_v1)
            if self.v2only and self.v1only:
                combined_dim = 2 * (self.d_v2 + self.d_v1)
            else:
                combined_dim = 2 * (self.d_p + self.d_v2 + self.d_v1)
        else:
            combined_dim = 2 * (self.d_p + self.d_v2 + self.d_v1)#300
        output_dim = hyp_params.output_dim       


        self.proj_p = nn.Conv1d(self.orig_d_p, self.d_p, kernel_size=1, padding=0, bias=False)
        self.proj_v2 = nn.Conv1d(self.orig_d_v2, self.d_v2, kernel_size=1, padding=0, bias=False)
        self.proj_v1 = nn.Conv1d(self.orig_d_v1, self.d_v1, kernel_size=1, padding=0, bias=False)

        
        if self.partial_mode == 2:
            if self.ponly and self.v2only:
                self.trans_p_with_v2 = self.get_network(self_type='pv2')
                self.trans_v2_with_p = self.get_network(self_type='v2p')
            elif self.ponly and self.v1only:
                self.trans_p_with_v1 = self.get_network(self_type='pv1')
                self.trans_v1_with_p = self.get_network(self_type='v1p')
            else:#
                self.trans_v2_with_v1 = self.get_network(self_type='v2v1')
                self.trans_v1_with_v2 = self.get_network(self_type='v1v2')
        if self.partial_mode == 3:
            self.trans_p_with_v2 = self.get_network(self_type='pv2')
            self.trans_p_with_v1 = self.get_network(self_type='pv1')
            self.trans_v2_with_p = self.get_network(self_type='v2p')
            self.trans_v2_with_v1 = self.get_network(self_type='v2v1')
            self.trans_v1_with_p = self.get_network(self_type='v1p')
            self.trans_v1_with_v2 = self.get_network(self_type='v1v2')
        if self.ponly:
            self.trans_p_mem = self.get_network(self_type=f'p_mem{self.partial_mode}', layers=3)
        if self.v2only:
            self.trans_v2_mem = self.get_network(self_type=f'v2_mem{self.partial_mode}', layers=3)
        if self.v1only:
            self.trans_v1_mem = self.get_network(self_type=f'v1_mem{self.partial_mode}', layers=3)
        
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
    def get_network(self, self_type='p', layers=-1):
        # if self_type == 'l':
        #     embed_dim, attn_dropout = self.d_p, self.attn_dropout
        # elif self_type == 'a':
        #     embed_dim, attn_dropout = self.d_v2, self.attn_dropout
        # elif self_type == 'v':
        #     embed_dim, attn_dropout = self.d_v1, self.attn_dropout
        # elif self_type in ['al','la']:
        #     embed_dim, attn_dropout = self.d_p, self.attn_dropout
        # elif self_type in ['lv','vl']:
        #     embed_dim, attn_dropout = self.d_p, self.attn_dropout
        # elif self_type in ['lv','vl']:
        #     embed_dim, attn_dropout = self.d_p, self.attn_dropout
        # print(self_type)
        if self_type in ['p', 'v2p', 'v1p']:
            embed_dim, attn_dropout = self.d_p, self.attn_dropout
        elif self_type in ['v2', 'pv2', 'v1v2']:
            embed_dim, attn_dropout = self.d_v2, self.attn_dropout_v2
        elif self_type in ['v1', 'pv1', 'v2v1']:
            embed_dim, attn_dropout = self.d_v1, self.attn_dropout_v1
        elif self_type == 'p_mem1':
            embed_dim, attn_dropout = self.d_p, self.attn_dropout
        elif self_type == 'v2_mem1':
            embed_dim, attn_dropout = self.d_v2, self.attn_dropout
        elif self_type == 'v1_mem1':
            embed_dim, attn_dropout = self.d_v1, self.attn_dropout
        elif self_type == 'p_mem2':
            embed_dim, attn_dropout = self.d_p, self.attn_dropout
        elif self_type == 'v2_mem2':
            embed_dim, attn_dropout = self.d_v2, self.attn_dropout
        elif self_type == 'v1_mem2':
            embed_dim, attn_dropout = self.d_v1, self.attn_dropout
        elif self_type == 'p_mem3':
            embed_dim, attn_dropout = 2*self.d_p, self.attn_dropout
        elif self_type == 'v2_mem3':
            embed_dim, attn_dropout = 2*self.d_v2, self.attn_dropout
        elif self_type == 'v1_mem3':
            embed_dim, attn_dropout = 2*self.d_v1, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self, x_p, x_v2, x_v1):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_p = F.dropout(x_p.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v2 = x_v2.transpose(1, 2)
        x_v1 = x_v1.transpose(1, 2)
        # print("shape:")
        # print(x_p.shape)
        # print(x_v2.shape)
        # print(x_v1.shape)
        # print(self.proj_v2)
        # self.d_p, self.d_v2, self.d_v1 = 30, 30, 30
        proj_x_p = x_p if self.orig_d_p == self.d_p else self.proj_p(x_p)
        proj_x_v2 = x_v2 if self.orig_d_v2 == self.d_v2 else self.proj_v2(x_v2)
        proj_x_v1 = x_v1 if self.orig_d_v1 == self.d_v1 else self.proj_v1(x_v1)
        # print("projected shape:")
        # print(proj_x_l.shape)
        # print(proj_x_a.shape)
        # print(proj_x_v.shape)
        proj_x_v2 = proj_x_v2.permute(2, 0, 1)
        proj_x_v1 = proj_x_v1.permute(2, 0, 1)
        proj_x_p = proj_x_p.permute(2, 0, 1)
        # print("final shape:")
        # print(proj_x_l.shape)
        # print(proj_x_a.shape)
        # print(proj_x_v.shape)
        if self.partial_mode == 1:
            if self.ponly:
                h_ps = self.trans_p_mem(proj_x_p)
                if type(h_ps) == tuple:
                    h_ps = h_ps[0]
                last_hs = h_ps[-1]#l pred
            elif self.v2only:
                h_v2s = self.trans_v2_mem(proj_x_v2)
                if type(h_v2s) == tuple:
                    h_v2s = h_v2s[0]
                last_hs = h_v2s[-1]#l pred
            else:
                h_v1s = self.trans_v1_mem(proj_x_v1)
                if type(h_v1s) == tuple:
                    h_v1s = h_v1s[0]
                last_hs = h_v1s[-1]#l pred
        
        if self.partial_mode == 2:
            if self.ponly and self.v2only:
                h_p_with_v2s = self.trans_p_with_v2(proj_x_p, proj_x_v2, proj_x_v2)
                h_v2_with_ps = self.trans_v2_with_p(proj_x_v2, proj_x_p, proj_x_p)
                
                h_ps = self.trans_p_mem(h_p_with_v2s)
                if type(h_ps) == tuple:
                    h_ps = h_ps[0]
                last_h_p = h_ps[-1]#l pred
                
                h_v2s = self.trans_v2_mem(h_v2_with_ps)
                if type(h_v2s) == tuple:
                    h_v2s = h_v2s[0]
                last_h_v2  = h_v2s[-1]#a pred
                
                last_hs = torch.cat([last_h_p, last_h_v2], dim=1)#prediction
                
            elif self.ponly and self.v1only:
                h_p_with_v1s = self.trans_p_with_v1(proj_x_p, proj_x_v1, proj_x_v1)
                h_v1_with_ps = self.trans_v1_with_p(proj_x_v1, proj_x_p, proj_x_p)
                
                h_ps = self.trans_p_mem(h_p_with_v1s)
                if type(h_ps) == tuple:
                    h_ps = h_ps[0]
                last_h_p = h_ps[-1]#l pred

                h_v1s = self.trans_v1_mem(h_v1_with_ps)
                if type(h_v1s) == tuple:
                    h_v1s = h_v1s[0]
                last_h_v1 = h_v1s[-1]#a pred
                
                last_hs = torch.cat([last_h_p, last_h_v1], dim=1)#prediction
                
            else:# a v
                h_v2_with_v1s = self.trans_v2_with_v1(proj_x_v2, proj_x_v1, proj_x_v1)
                h_v1_with_v2s = self.trans_v1_with_v2(proj_x_v1, proj_x_v2, proj_x_v2)
                
                h_v2s = self.trans_v2_mem(h_v2_with_v1s)
                if type(h_v2s) == tuple:
                    h_v2s = h_v2s[0]
                last_h_v2 = h_v2s[-1]#l pred
                
                h_v1s = self.trans_v1_mem(h_v1_with_v2s)
                if type(h_v1s) == tuple:
                    h_v1s = h_v1s[0]
                last_h_v1 = h_v1s[-1]#a pred

                last_hs = torch.cat([last_h_v2, last_h_v1], dim=1)#prediction
        
        if self.partial_mode == 3:
            h_p_with_v2s = self.trans_p_with_v2(proj_x_p, proj_x_v2, proj_x_v2)
            h_p_with_v1s = self.trans_p_with_v1(proj_x_p, proj_x_v1, proj_x_v1)
            h_ps = torch.cat([h_p_with_v2s, h_p_with_v1s], dim=2)
            h_ps = self.trans_p_mem(h_ps)
            if type(h_ps) == tuple:
                h_ps = h_ps[0]
            last_h_p = h_ps[-1]
            
            h_v2_with_ps = self.trans_v2_with_p(proj_x_v2, proj_x_p, proj_x_p)
            h_v2_with_v1s = self.trans_v2_with_v1(proj_x_v2, proj_x_v1, proj_x_v1)
            h_v2s = torch.cat([h_v2_with_ps, h_v2_with_v1s], dim=2)
            h_v2s = self.trans_v2_mem(h_v2s)
            if type(h_v2s) == tuple:
                h_v2s = h_v2s[0]
            last_h_v2 = h_v2s[-1]
            
            h_v1_with_ps = self.trans_v1_with_p(proj_x_v1, proj_x_p, proj_x_p)
            h_v1_with_v2s = self.trans_v1_with_v2(proj_x_v1, proj_x_v2, proj_x_v2)
            h_v1s = torch.cat([h_v1_with_ps, h_v1_with_v2s], dim=2)
            h_v1s = self.trans_v1_mem(h_v1s)
            if type(h_v1s) == tuple:
                h_v1s = h_v1s[0]
            last_h_v1 = h_v1s[-1]
            
            last_hs = torch.cat([last_h_p, last_h_v2, last_h_v1], dim=1)
        # print(last_hs.shape)
        # print(self.proj1)
        # print(self.proj1(last_hs).shape)
        # print(F.relu(self.proj1(last_hs)).shape)
        # print(F.dropout(F.relu(self.proj1(last_hs))).shape)
        # print(self.proj2)
        # print(self.proj2(F.dropout(F.relu(self.proj1(last_hs)))).shape)
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output, last_hs
