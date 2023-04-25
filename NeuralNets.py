# Includes the neural nets

import torch.nn as nn
import torch


class SimpleLSTM(nn.Module):
    '''
    The simplest neural nets
    The structure likes this:

    LSTM layers -> Drop out layer -> Linear layer
    
    
    '''
    def __init__(self, input_dim = 150, hidden_dim = 64, output_dim =1, lstm_nums_layer =2 , dropout = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        
        self.LSTM = nn.LSTM(input_dim, hidden_dim, lstm_nums_layer, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)

        self.to(torch.double)
        
    def forward(self, input):
        # print(input.shape)
        
        hidden_state, _ = self.LSTM(input)
        
        # print(hidden_state.shape)
        output = self.dropout(hidden_state)
        output = self.hidden_to_output(output)
        
        
        return output


# class SimpleLSTM(nn.Module):
#     '''
#     The simplest neural nets
#     The structure likes this:

#     LayerNorm -> LSTM layers -> Drop out layer -> Linear layer

#     '''
#     def __init__(self, input_dim = 150, hidden_dim = 64, output_dim = 1, lstm_nums_layer = 2, dropout = 0.1):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim

#         self.layer_norm = nn.LayerNorm(input_dim)
#         self.LSTM = nn.LSTM(input_dim, hidden_dim, lstm_nums_layer, batch_first = True)
#         self.dropout = nn.Dropout(dropout)

#         self.hidden_to_output = nn.Linear(hidden_dim, output_dim)

#         self.to(torch.double)

#     def forward(self, input):
#         # print(input.shape)

#         input = self.layer_norm(input)
#         hidden_state, _ = self.LSTM(input)

#         # print(hidden_state.shape)
#         output = self.dropout(hidden_state)
#         output = self.hidden_to_output(output)

#         return output



import torch
import torch.nn as nn

# class SimpleLSTM(nn.Module):
#     '''
#     The simplest neural nets
#     The structure likes this:

#     BatchNorm -> LSTM layers -> Drop out layer -> Linear layer

#     '''
#     def __init__(self, input_dim = 150, hidden_dim = 64, output_dim = 1, lstm_nums_layer = 2, dropout = 0.1):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim

#         self.batch_norm = nn.BatchNorm1d(input_dim)
#         self.LSTM = nn.LSTM(input_dim, hidden_dim, lstm_nums_layer, batch_first = True)
#         self.dropout = nn.Dropout(dropout)

#         self.hidden_to_output = nn.Linear(hidden_dim, output_dim)

#         self.to(torch.double)

#     def forward(self, input):
#         # print(input.shape)

#         # Permute the input tensor to apply batch normalization along the time dimension
#         input = input.permute(0, 2, 1)
#         input = self.batch_norm(input)
#         input = input.permute(0, 2, 1)

#         hidden_state, _ = self.LSTM(input)

#         # print(hidden_state.shape)
#         output = self.dropout(hidden_state)
#         output = self.hidden_to_output(output)

#         return output