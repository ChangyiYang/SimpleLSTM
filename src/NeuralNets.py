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


class SimpleAttentionLSTM(nn.Module):
    '''
    The simplest neural nets
    The structure likes this:

    Linear layer -> Attention layer -> Dropout layer -> Linear layer
    '''
    def __init__(self, input_dim=150, hidden_dim=64, output_dim=1, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1)
        self.dropout = nn.Dropout(dropout)
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)

        self.to(torch.double)
        
    def forward(self, input):
        output = self.linear(input)
        output, _ = self.attention(output, output, output)  # Apply attention
        output = self.dropout(output)
        output = self.hidden_to_output(output)
        
        return output


class DimensionReductionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lstm_num_layers, dropout):
        super(DimensionReductionModel, self).__init()
        
        # Add more crossed linear and non-linear layers for dimension reduction
        self.linear1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(64, hidden_dim)
        self.relu3 = nn.ReLU()

        self.simple_lstm = SimpleLSTM(hidden_dim, hidden_dim, output_dim, lstm_num_layers, dropout)

    def forward(self, input):
        # Apply additional crossed linear and non-linear layers
        x = self.linear1(input)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)

        # Pass the reduced dimension data to the SimpleLSTM
        output = self.simple_lstm(x)

        return output

