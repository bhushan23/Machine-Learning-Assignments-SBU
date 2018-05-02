
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
        super(SequenceClassify, self).__init__()
        
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

