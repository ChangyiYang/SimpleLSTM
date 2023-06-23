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
        
        # tried to add layer norm or batch norm here, but failed
        # these two norm layers do not normalize the keff, which oscillate very small, so the predicting result is just a striaght line. 

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


