import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn= nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.embed(features))
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=0.4, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        # self.init_weights()
    
    def init_weights(self):
        self.word_embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
            
    
    def forward(self, features, captions):
        # print('feature size {}'.format(features.size()))
        # print('caption size {}'.format(captions.size()))
        # feature_size = batch_size * 256
        # caption_size = batch_size * caption_length
        # exclude the <end> token
        # embed_matrix_size = batch_size * caption_size * embed_dim
        embed_matrix  = self.word_embed(captions[:, :-1])
        # print('embed_matrix size {}'.format(embed_matrix.size()))
        # concatenate the features and embed by stacking them horizontally
        # sequence_length = batch_size * caption_length * embed_dim
        # print('feautre unsqueeze size {}'.format(features.unsqueeze(1).size()))
        sequence = torch.cat((features.unsqueeze(1), embed_matrix), 1)
        # print('input seq size {}'.format(sequence.size()))
        # lstm_1_output_size = batch_size* caption_length * hidden_size
        # h0 = torch.nn.init_xavier(-1, 1)
        # c0  = torch.nn.init_xavier(-1,1)
        lstm_1_output, _ = self.lstm(sequence)
        # print('lstm_1_output size {}'.format(lstm_1_output.size()))
        # ouptut_size = batch_size * vocab_size
        output = self.linear(lstm_1_output)
        # print('output size {}'.format(output.size()))
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pred_sentences = []
        for i in range(max_len):
            hidden, states = self.lstm(inputs, states)
            output = self.linear(hidden.squeeze(1))
            _, pred = output.max(1)
            pred_sentences.append(pred.item())
            inputs = self.word_embed(pred).unsqueeze(1)
        return pred_sentences
            