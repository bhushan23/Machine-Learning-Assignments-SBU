import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as DD
import torchnet as tnt

class SequenceClassify(nn.Module):
    def __init__(self):
        super(SequenceClassify, self).__init__()
        
        ############## 1st To Do (20 points) ##############
        ###################################################
        self.recurrent_layer = nn.LSTM(hidden_size = 100, input_size = 75)
        self.project_layer = nn.Linear(100, 10)
        ###################################################
    
    # the size of input is [batch_size, seq_len(15), input_dim(75)]
    # the size of logits is [batch_size, num_class]
    def forward(self, input, h_t_1=None, c_t_1=None):
        # the size of rnn_outputs is [batch_size, seq_len, rnn_size]
        rnn_outputs, (hn, cn) = self.recurrent_layer(input)
        # classify the last step of rnn_outpus
        # the size of logits is [batch_size, num_class]
        logits = self.project_layer(rnn_outputs[:,-1])
        return logits

# sequence classification model
class M1(nn.Module):
    def __init__(self):
        super(M1, self).__init__()
        
        self.recurrent_layer  = nn.LSTM(hidden_size = 100, input_size = 75, num_layers = 5)
        self.recurrent_layer1  = nn.LSTM(hidden_size = 200, input_size = 100, num_layers = 5)
        self.recurrent_layer2  = nn.LSTM(hidden_size = 300, input_size = 200, num_layers = 5)
        self.project_layer     = nn.Linear(300, 200)
        self.project_layer1    = nn.Linear(200, 100)
        self.project_layer2    = nn.Linear(100, 10)
    
    # the size of input is [batch_size, seq_len(15), input_dim(75)]
    # the size of logits is [batch_size, num_class]
    def forward(self, input, h_t_1=None, c_t_1=None):
        # the size of rnn_outputs is [batch_size, seq_len, rnn_size]
        # self.recurrent_layer.flatten_parameters()
        rnn_outputs, (hn, cn) = self.recurrent_layer(input)
        rnn_outputs, (hn, cn) = self.recurrent_layer1(rnn_outputs)
        rnn_outputs, (hn, cn) = self.recurrent_layer2(rnn_outputs)
        # classify the last step of rnn_outpus
        # the size of logits is [batch_size, num_class]
        logits = self.project_layer(rnn_outputs[:,-1])
        logits = self.project_layer1(logits)
        logits = self.project_layer2(logits)
        return logits

class M2(nn.Module):
    def __init__(self):
        super(M2, self).__init__()
        
        self.recurrent_layer  = nn.LSTM(hidden_size = 100, input_size = 75, num_layers = 1)
        self.bNorm            = nn.BatchNorm1d(15)
        self.recurrent_layer1  = nn.LSTM(hidden_size = 100, input_size = 100, num_layers = 1)
        self.project_layer   = nn.Linear(100, 10)
    
    # the size of input is [batch_size, seq_len(15), input_dim(75)]
    # the size of logits is [batch_size, num_class]
    def forward(self, input, h_t_1=None, c_t_1=None):
        # the size of rnn_outputs is [batch_size, seq_len, rnn_size]
        # self.recurrent_layer.flatten_parameters()
        x, (hn, cn) = self.recurrent_layer(input)
        x = self.bNorm(x)
        x = x[:,-1]
        x = x.contiguous() #.view(:, -1)
        # classify the last step of rnn_outpus
        # the size of logits is [batch_size, num_class]
        logits = self.project_layer(x)
        return logits


class M3(nn.Module):
    def __init__(self):
        super(M3, self).__init__()
        
        self.recurrent_layer  = nn.LSTM(hidden_size = 100, input_size = 75, num_layers = 1)
        self.bNorm            = nn.BatchNorm1d(15)
        self.recurrent_layer1  = nn.LSTM(hidden_size = 100, input_size = 100, num_layers = 1)
        self.project_layer   = nn.Linear(100, 50)
        self.project_layer2   = nn.Linear(50, 10)
        self.bNorm1           = nn.BatchNorm1d(50)
        self.dropout          = nn.Dropout2d(p = 0.0003)
        self.relu             = nn.ReLU() 

    # the size of input is [batch_size, seq_len(15), input_dim(75)]
    # the size of logits is [batch_size, num_class]
    def forward(self, input, h_t_1=None, c_t_1=None):
        # the size of rnn_outputs is [batch_size, seq_len, rnn_size]
        # self.recurrent_layer.flatten_parameters()
        # x = self.bNorm(input)
        x, (hn, cn) = self.recurrent_layer(input)
        # print(input.size(), x.size())
        x = self.bNorm(x)
        # print('X Size: ',x.size())
        x = x[:,-1]
        x = x.contiguous() #.view(:, -1)
        # classify the last step of rnn_outpus
        # the size of logits is [batch_size, num_class]
        logits = self.project_layer(x)
        logits = self.dropout(logits)
        # print(logits.size())
        logits = self.bNorm1(logits)
        logits = self.relu(logits)
        logits = self.project_layer2(logits)
        return logits


class M4(nn.Module):
    def __init__(self):
        super(M4, self).__init__()
        
        self.recurrent_layer  = nn.LSTM(hidden_size = 100, input_size = 75, num_layers = 1)
        self.bNorm            = nn.BatchNorm1d(15)
        self.project_layer    = nn.Linear(100, 200)
        self.project_layer1   = nn.Linear(200, 400)
        self.project_layer2   = nn.Linear(400, 200)
        self.project_layer3    = nn.Linear(200, 100)
        self.project_layer4   = nn.Linear(100, 10)
        self.bNorm1           = nn.BatchNorm1d(50)
        self.bNorm2 = nn.BatchNorm1d(100)
        self.bNorm3 = nn.BatchNorm1d(200)
        self.bNorm4 = nn.BatchNorm1d(400)
        self.dropout          = nn.Dropout2d(p = 0.2)
        self.relu             = nn.ReLU() 
    # the size of input is [batch_size, seq_len(15), input_dim(75)]
    # the size of logits is [batch_size, num_class]
    def forward(self, input, h_t_1=None, c_t_1=None):
        # the size of rnn_outputs is [batch_size, seq_len, rnn_size]
        # self.recurrent_layer.flatten_parameters()
        # x = self.bNorm(input)
        x, (hn, cn) = self.recurrent_layer(input)
        # print(input.size(), x.size())
        x = self.bNorm(x)
        #print('X Size: ',x.size())
        l = x[:,-1]
        # x = x.contiguous() #.view(:, -1)
        # classify the last step of rnn_outpus
        # the size of logits is [batch_size, num_class]
        #  print(x.size())
        logits = self.project_layer(l)
        logits = self.bNorm3(logits)
        logits = self.relu(logits)
        logits = self.dropout(logits)
        logits = self.project_layer1(logits)
        logits = self.bNorm4(logits)
        logits = self.relu(logits)
        logits = self.dropout(logits)
        logits = self.project_layer2(logits)
        logits = self.bNorm3(logits)
        logits = self.relu(logits)
        logits = self.dropout(logits)
        logits = self.project_layer3(logits)
        logits = self.bNorm2(logits)
        logits = self.relu(logits)
        logits = self.dropout(logits)
        logits = self.project_layer4(logits)
        return logits



class M5(nn.Module):
    def __init__(self):
        super(M5, self).__init__()
        self.recurrent_layer  = nn.LSTM(hidden_size = 100, batch_first = True, input_size = 73, num_layers = 2)
        self.bNorm            = nn.BatchNorm1d(200)
        self.bNorm100            = nn.BatchNorm1d(100)

        self.recurrent_layer1  = nn.LSTM(hidden_size = 100, input_size = 100, num_layers = 1)
        self.project_layer   = nn.Linear(100, 50)
        self.project_layer2   = nn.Linear(50, 10)
        self.bNorm1           = nn.BatchNorm1d(50)
        self.dropout          = nn.Dropout2d(p = 0.3)
        self.conv1            = nn.Conv1d(15, 200, 2) 
        self.conv2            = nn.Conv1d(200, 100, 2) 
        self.relu             = nn.ReLU() 

    # the size of input is [batch_size, seq_len(15), input_dim(75)]
    # the size of logits is [batch_size, num_class]
    def forward(self, input, h_t_1=None, c_t_1=None):
        # the size of rnn_outputs is [batch_size, seq_len, rnn_size]
        # self.recurrent_layer.flatten_parameters()
        # x = self.bNorm(input)
        x = self.conv1(input)
        # print(x.size())
        x = self.relu(x)
        x = self.bNorm(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bNorm100(x)
        x = self.dropout(x)

        x, (hn, cn) = self.recurrent_layer(x)
        # print(input.size(), x.size())
        x = self.bNorm100(x)
        # print('X Size: ',x.size())
        x = x[:,-1]
        x = x.contiguous() #.view(:, -1)
        # classify the last step of rnn_outpus
        # the size of logits is [batch_size, num_class]
        logits = self.project_layer(x)
        logits = self.dropout(logits)
        # print(logits.size())
        logits = self.bNorm1(logits)
        logits = self.project_layer2(logits)
        return logits


class M6(nn.Module):
    def __init__(self):
        super(M6, self).__init__()
        self.recurrent_layer  = nn.LSTM(hidden_size = 100, batch_first = True, input_size = 74, num_layers = 2)
        self.bNorm            = nn.BatchNorm1d(200)
        self.bNorm100            = nn.BatchNorm1d(100)

        self.recurrent_layer1  = nn.LSTM(hidden_size = 100, input_size = 100, num_layers = 1)
        self.project_layer   = nn.Linear(99, 50)
        self.project_layer2   = nn.Linear(50, 10)
        self.bNorm1           = nn.BatchNorm1d(50)
        self.dropout          = nn.Dropout2d(p = 0.3)
        self.conv1            = nn.Conv1d(15, 200, 2) 
        self.conv2            = nn.Conv1d(200, 100, 2) 
        self.relu             = nn.ReLU() 

    # the size of input is [batch_size, seq_len(15), input_dim(75)]
    # the size of logits is [batch_size, num_class]
    def forward(self, input, h_t_1=None, c_t_1=None):
        # the size of rnn_outputs is [batch_size, seq_len, rnn_size]
        # self.recurrent_layer.flatten_parameters()
        # x = self.bNorm(input)
        x = self.conv1(input)
        # print(x.size())
        x = self.relu(x)
        x = self.bNorm(x)
        x = self.dropout(x)
        x, (hn, cn) = self.recurrent_layer(x)
        x = self.conv2(x)
        # print(x.size())
        x = self.relu(x)
        # print(input.size(), x.size())
        x = self.bNorm100(x)
        x = self.dropout(x)
        # print('X Size: ',x.size())
        x = x[:,-1]
        x = x.contiguous() #.view(:, -1)
        # classify the last step of rnn_outpus
        # the size of logits is [batch_size, num_class]
        logits = self.project_layer(x)
        logits = self.dropout(logits)
        # print(logits.size())
        logits = self.bNorm1(logits)
        logits = self.project_layer2(logits)
        return logits

class M7(nn.Module):
    def __init__(self):
        super(M7, self).__init__()
        self.recurrent_layer  = nn.LSTM(hidden_size = 100, batch_first = True, input_size = 74, num_layers = 2)
        self.bNorm            = nn.BatchNorm1d(200)
        self.bNorm100            = nn.BatchNorm1d(100)

        self.recurrent_layer1  = nn.LSTM(hidden_size = 100, input_size = 100, num_layers = 2)
        self.project_layer   = nn.Linear(99, 50)
        self.project_layer2   = nn.Linear(50, 10)
        self.bNorm1           = nn.BatchNorm1d(50)
        self.dropout          = nn.Dropout2d(p = 0.2)
        self.conv1            = nn.Conv1d(15, 200, 2) 
        self.conv2            = nn.Conv1d(200, 100, 2) 
        self.relu             = nn.ReLU() 

    # the size of input is [batch_size, seq_len(15), input_dim(75)]
    # the size of logits is [batch_size, num_class]
    def forward(self, input, h_t_1=None, c_t_1=None):
        # the size of rnn_outputs is [batch_size, seq_len, rnn_size]
        # self.recurrent_layer.flatten_parameters()
        # x = self.bNorm(input)
        x = self.conv1(input)
        # print(x.size())
        #x = self.relu(x)
        x = self.bNorm(x)
        x = self.dropout(x)
        x, (hn, cn) = self.recurrent_layer(x)
        x = self.conv2(x)
        # print(x.size())
        # x = self.relu(x)
        # print(input.size(), x.size())
        x = self.bNorm100(x)
        x = self.dropout(x)
        # print('X Size: ',x.size())
        x = x[:,-1]
        x = x.contiguous() #.view(:, -1)
        # classify the last step of rnn_outpus
        # the size of logits is [batch_size, num_class]
        logits = self.project_layer(x)
        logits = self.dropout(logits)
        # print(logits.size())
        logits = self.bNorm1(logits)
        #logits = self.relu(logits)
        logits = self.project_layer2(logits)
        return logits


class M8(nn.Module):
    def __init__(self):
        super(M8, self).__init__()
        self.recurrent_layer  = nn.LSTM(hidden_size = 100, batch_first = True, input_size = 73, num_layers = 2)
        self.bNorm            = nn.BatchNorm1d(200)
        self.bNorm100            = nn.BatchNorm1d(100)

        self.project_layer   = nn.Linear(100, 50)
        self.project_layer2   = nn.Linear(50, 10)
        self.bNorm1           = nn.BatchNorm1d(50)
        self.dropout          = nn.Dropout2d(p = 0.4)
        self.conv1            = nn.Conv1d(15, 200, 2) 
        self.conv2            = nn.Conv1d(200, 100, 2) 
        self.conv3           = nn.Conv1d(200, 100, 2) 
        self.relu             = nn.ReLU() 

    # the size of input is [batch_size, seq_len(15), input_dim(75)]
    # the size of logits is [batch_size, num_class]
    def forward(self, input, h_t_1=None, c_t_1=None):
        # the size of rnn_outputs is [batch_size, seq_len, rnn_size]
        # self.recurrent_layer.flatten_parameters()
        # x = self.bNorm(input)
        x = self.conv1(input)
        # print(x.size())
        x = self.relu(x)
        x = self.bNorm(x)
        x = self.dropout(x)
        x = self.conv3(x)
        # print(x.size())
        x = self.relu(x)
        x = self.bNorm100(x)
        x = self.dropout(x)
        x, (hn, cn) = self.recurrent_layer(x)
        #x = self.conv2(x)
        # print(x.size())
        #x = self.relu(x)
        # print(input.size(), x.size())
        #x = self.bNorm100(x)
        #x = self.dropout(x)
        #print('X Size: ',x.size())
        x = x[:,-1]
        #x = x.contiguous() #.view(:, -1)
        # classify the last step of rnn_outpus
        # the size of logits is [batch_size, num_class]
        logits = self.project_layer(x)
        logits = self.dropout(logits)
        # print(logits.size())
        logits = self.bNorm1(logits)
        logits = self.project_layer2(logits)
        return logits


