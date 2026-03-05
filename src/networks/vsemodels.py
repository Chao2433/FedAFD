
import os
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models
import torchtext
import torch


def get_vsemodel(word2idx, config, mlp_local):
    return VSEPP(word2idx, config, mlp_local)


class VSEPP(nn.Module):
    def __init__(self, word2idx, config, mlp_local):
        super(VSEPP, self).__init__()
        self.image_extractor = ImageRepExtractor(config, mlp_local)
        self.text_extractor = TextRepExtractor(word2idx, config, mlp_local)

    def forward(self, images, captions, cap_lens):
        sorted_cap_len, sorted_cap_indices = torch.sort(cap_lens, 0, True)
        images = images[sorted_cap_indices]
        captions = captions[sorted_cap_indices]
        cap_lens = sorted_cap_len
        image_code = self.image_extractor(images)
        text_code = self.text_extractor(captions, cap_lens)

        if not self.train:
            _, recover_indices = torch.sort(sorted_cap_indices)
            image_code = image_code[recover_indices]
            text_code = text_code[recover_indices]

        return image_code, text_code


class ImageRepExtractor(nn.Module):
    def __init__(self, config, mlp_local):
        super(ImageRepExtractor, self).__init__()
        embed_dim = config.embed_dim

        net = self.cnn = getattr(models, config.cnn_type)(pretrained=True)
        cnn_dim = self.cnn_dim = self.cnn.fc.in_features

        for idx, param in enumerate(self.cnn.parameters()):
            param.requires_grad = True

        self.fc = nn.Linear(cnn_dim, embed_dim)
        self.cnn.fc = nn.Sequential()

        self.net = net

    def forward(self, x):
        out = self.net(x)
        out = nn.functional.normalize(out)
        return out

class TextRepExtractor(nn.Module):
    def __init__(self,word2idx, opt, mlp_local):

        super(TextRepExtractor, self).__init__()
        wemb_type, word_dim, embed_dim = \
            opt.wemb_type, opt.word_dim, opt.embed_dim
        self.embed = nn.Embedding(len(word2idx), word_dim)
        self.embed.weight.requires_grad = True
        self.rnn = nn.GRU(word_dim, embed_dim // 2, bidirectional=True, batch_first=True)

        self.init_weights(wemb_type, word2idx, word_dim, opt.cache_dir)

    def init_weights(self, wemb_type, word2idx, word_dim, cache_dir=os.environ['HOME'] + '/data/mmdata'):
        if wemb_type is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText(cache='data')
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache='data')
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        x = self.embed(x)
        packed = pack_padded_sequence(input=x, lengths=lengths, batch_first=True)
        _, hidden = self.rnn(packed)
        out = nn.functional.normalize(hidden[-1])
        return out