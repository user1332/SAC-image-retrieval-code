import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F 
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import torchvision
from torch.utils import model_zoo


class SA_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(SA_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        attention = F.dropout(attention, p=0.1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))


#         rnn_output=rnn_output.reshape(-1,2*self.in_dim_text)
#         output = self.out_linear(rnn_output)
        output = self.out_linear(h_n.squeeze(0))
        return output

class ImageEncoder_MulGate_3_2Seq_SA_multiple_sentence_attention_map_soft_multiple(nn.Module):
    def __init__(self, embed_size, model='50'):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ImageEncoder_MulGate_3_2Seq_SA_multiple_sentence_attention_map_soft_multiple, self).__init__()
        resnet_50 = models.resnet50(pretrained=True)
        self.in_dim_text = 512
        embed_dim = 512
        self.low_level_texture = nn.Sequential(
                        resnet_50.conv1,
                        resnet_50.bn1,
                        resnet_50.relu,
                        resnet_50.maxpool,
                        resnet_50.relu,
                        resnet_50.layer1,
                        resnet_50.layer2
        )
        self.mid_level_texture = resnet_50.layer3


        self.avgpool_low = GlobalAvgPool2d()
        self.avgpool_mid = GlobalAvgPool2d()
        self.fc_low = torch.nn.Linear(512, embed_dim)
        self.fc_mid = torch.nn.Linear(1024, embed_dim)
        self.bn_low = nn.BatchNorm1d(512, momentum=0.01)
        self.bn_mid = nn.BatchNorm1d(512, momentum=0.01)
        self.bn_rnn = nn.BatchNorm1d(512, momentum=0.01)
        self.rnn = nn.LSTM(         
            input_size=512,
            hidden_size=512,         # rnn hidden unit
            num_layers=1,           
            batch_first=True,
                )
    
        self.sa_low = SA_Module(512)
        self.sa_mid = SA_Module(1024)
#         self.wa_low = sentence_attention_map(512,512)
#         self.wa_mid = sentence_attention_map(512,1024)
        self.softmax = nn.Softmax(dim=-1)
        self.out_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.gamma_low = nn.Parameter(torch.zeros(1))
        self.gamma_mid = nn.Parameter(torch.zeros(1))
#         self.gamma_word_low = nn.Parameter(torch.zeros(1))
#         self.gamma_word_mid = nn.Parameter(torch.zeros(1))
        self.init_weights()
    
    def init_weights(self):
        print ("Initializing image encoder fc")
        r = np.sqrt(6.) / np.sqrt(self.fc_low.in_features + self.fc_low.out_features)
        self.fc_low.weight.data.uniform_(-r, r)
        self.fc_low.bias.data.fill_(0)
        r = np.sqrt(6.) / np.sqrt(self.fc_mid.in_features + self.fc_mid.out_features)
        self.fc_mid.weight.data.uniform_(-r, r)
        self.fc_mid.bias.data.fill_(0)
        r = np.sqrt(6.) / np.sqrt(self.out_linear.in_features + self.out_linear.out_features)
        self.out_linear.weight.data.uniform_(-r, r)
#         self.out_linear.bias.data.fill_(0)
        
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative

    def get_trainable_parameters(self):
        return list(self.parameters())

    def forward(self, image,epoch=0):
        
        if (epoch<=10):
            with torch.no_grad():
                out_low = self.low_level_texture(image)
                out_mid = self.mid_level_texture(out_low)
        else:
            out_low = self.low_level_texture(image)
            out_mid = self.mid_level_texture(out_low)          
        fc_low = self.bn_low(self.fc_low(self.gem(out_low).squeeze(-1).squeeze(-1))).unsqueeze(1)
        fc_mid = self.bn_mid(self.fc_mid(self.gem(out_mid).squeeze(-1).squeeze(-1))).unsqueeze(1)
        rnn_input = torch.cat([fc_low, fc_mid], 1)
        rnn_output, (h_n,c_n) = self.rnn(rnn_input, None)
        output = self.out_linear(h_n.squeeze(0))
        return output

    def forward_compose(self, image , text_low,text_mid , epoch):
        if (epoch<=10):
            with torch.no_grad():
                out_low = self.low_level_texture(image)
                out_mid = self.mid_level_texture(out_low)
        else:
            out_low = self.low_level_texture(image)
            out_mid = self.mid_level_texture(out_low)

        bs = image.size(0)
        text_low = text_low.view(-1, self.in_dim_text, 1, 1).repeat(1, 1, out_low.size(2), out_low.size(3))
        text_low = text_low.view(bs, -1, out_low.size(2) * out_low.size(3)).permute(0, 2, 1)
        
#         out_low_split = torch.split(out_low, 512//2, 1)
#         out_low_sa = []
#         for split in out_low_split:
#             out_low_sa.append(self.sa_low(split))
            
#         out_low_sa = torch.cat(out_low_sa, dim=1)
        out_low_sa = self.sa_low(out_low)
        
        out_low_key = out_low_sa.view(bs, -1, out_low_sa.size(2) * out_low_sa.size(3))
        low_energy = torch.bmm(text_low, out_low_key)
        attention_low = self.softmax(low_energy) #change sigmoid to softmax
        low_val = out_low_sa.view(bs, -1, out_low_sa.size(2) * out_low_sa.size(3))
        attention_output_low = torch.bmm(low_val, attention_low.permute(0, 2, 1)).view(bs, -1, out_low_sa.size(2), out_low_sa.size(3))
#         out_low_wa = self.wa_low(out_low_sa,sen1,sen2)
        out_low_composed = self.gamma_low * attention_output_low + out_low_sa 
#         out_low_composed = out_low_sa + self.gamma_word_low*out_low_wa
#         out_low_composed = out_low
        
        text_mid = text_mid.view(-1, self.in_dim_text, 1, 1).repeat(1, 1, out_mid.size(2), out_mid.size(3))
        text_mid = text_mid.view(bs, -1, out_mid.size(2) * out_mid.size(3)).permute(0, 2, 1)
        out_mid_sa = self.sa_mid(out_mid)
#)         out_mid_split = torch.split(out_mid, 1024//2, 1)
#         out_mid_sa = []
#         for split in out_mid_split:
#             out_mid_sa.append(self.sa_mid(split))
#         out_mid_sa = torch.cat(out_mid_sa, dim=1
        out_mid_key = out_mid_sa.view(bs, -1, out_mid_sa.size(2) * out_mid_sa.size(3))

        mid_energy = torch.bmm(text_mid, out_mid_key)
        attention_mid = self.softmax(mid_energy)  # sig -> soft
        mid_val = out_mid_sa.view(bs, -1, out_mid_sa.size(2) * out_mid_sa.size(3))
        attention_output_mid = torch.bmm(mid_val, attention_mid.permute(0, 2, 1)).view(bs, -1, out_mid_sa.size(2), out_mid_sa.size(3))

        out_mid_composed = self.gamma_mid * attention_output_mid + out_mid_sa 

        fc_low = self.bn_low(self.fc_low(self.gem(out_low_composed).squeeze(-1).squeeze(-1))).unsqueeze(1)
        fc_mid = self.bn_mid(self.fc_mid(self.gem(out_mid_composed).squeeze(-1).squeeze(-1))).unsqueeze(1)
        rnn_input = torch.cat([fc_low, fc_mid], 1)
        rnn_output, (h_n,c_n) = self.rnn(rnn_input, None)
        output = self.out_linear(self.bn_rnn(h_n.squeeze(0)))
        return output
    
    
    
class DummyCaptionEncoder_without_embed_multiple_random_embeddings(nn.Module):
    def __init__(self, vocab_size, vocab_embed_size, embed_size):
        
        super(DummyCaptionEncoder_without_embed_multiple_random_embeddings, self).__init__()
        
        print (vocab_size,vocab_embed_size,embed_size)
        self.embed= nn.Embedding(vocab_size,768).cuda()
        self.out_linear = nn.Linear(1024,512, bias=False)
        self.out_linear_low = nn.Linear(1024,512, bias=False)
        self.out_linear_mid = nn.Linear(1024,1024, bias=False)
        self.rnn = nn.GRU(768,1024)
        self.init_weights()
        
    def forward(self, input, lengths):
#         print(min(input), max(input))
        input = self.embed(input)
        lengths = lengths.cpu()
        [_, sort_ids] = torch.sort(lengths, descending=True)
        sorted_input = input[sort_ids]
        sorted_length = lengths[sort_ids]
        reverse_sort_ids = sort_ids.clone()

        for i in range(sort_ids.size(0)):
            reverse_sort_ids[sort_ids[i]] = i
        packed = pack_padded_sequence(sorted_input, sorted_length, batch_first=True)
        self.rnn.flatten_parameters()
        output, _ = self.rnn(packed)

        padded, output_length = torch.nn.utils.rnn.pad_packed_sequence(output)
        
        output = [padded[output_length[i]-1, i, :] for i in range(len(output_length))]
        output = torch.stack([output[reverse_sort_ids[i]] for i in range(len(output))], dim=0)
        output_global = self.out_linear(output)
        output_low = self.out_linear_low(output)
        output_mid = self.out_linear_mid(output)
        return output_low,output_mid,output_global

    def get_trainable_parameters(self):
        return list(self.parameters())
    
    def init_weights(self):
        print ("Initializing caption encoder fc")
        r = np.sqrt(6.) / np.sqrt(self.out_linear_low.in_features + self.out_linear_low.out_features)
        self.out_linear_low.weight.data.uniform_(-r, r)
        r = np.sqrt(6.) / np.sqrt(self.out_linear_mid.in_features + self.out_linear_mid.out_features)
        self.out_linear_mid.weight.data.uniform_(-r, r)
        r = np.sqrt(6.) / np.sqrt(self.out_linear.in_features + self.out_linear.out_features)
        self.out_linear.weight.data.uniform_(-r, r)