import torch
import torch.nn as nn


# RGCN - river-dl version
class RGCN_v0(nn.Module):
    # Built off of https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
    def __init__(self, input_dim, hidden_dim, adj_matrix, recur_dropout = 0, dropout = 0, DA=False, device='cpu'):
        super().__init__()
        
        # New stuff
        self.A = torch.from_numpy(adj_matrix).float().to(device)
        
        self.W_graph_h = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_graph_h = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_graph_c = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_graph_c = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_h_cur = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.W_h_prev = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_h = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_c_cur = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.W_c_prev = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_c = nn.Parameter(torch.Tensor(hidden_dim))
        # End of new stuff
        
        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.weight_ih = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.init_weights()
        
        self.dropout = nn.Dropout(dropout)
        self.recur_dropout = nn.Dropout(recur_dropout)
        
        self.dense = nn.Linear(hidden_dim, 1)
        self.DA = DA
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        
    def forward(self, x, init_states = None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
        
        x = self.dropout(x)
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t_cur = f_t * c_t + i_t * self.recur_dropout(g_t)
            h_t_cur = o_t * torch.tanh(c_t)
            
            h_graph_t = torch.tanh(self.A @ (h_t @ self.W_graph_h + self.b_graph_h))
            c_graph_t = torch.tanh(self.A @ (c_t @ self.W_graph_c + self.b_graph_c))
            
            # aka h_update_t / c_update_t
            h_t = torch.sigmoid(h_t_cur @ self.W_h_cur + h_graph_t @ self.W_h_prev + self.b_h)
            c_t = torch.sigmoid(c_t_cur @ self.W_c_cur + c_graph_t @ self.W_c_prev + self.b_c) 
            
            hidden_seq.append(h_t.unsqueeze(1))
        hidden_seq = torch.cat(hidden_seq, dim= 1)
        out = self.dense(hidden_seq)
        if self.DA:
            return out, (h_t, c_t)
        else:
            return out