from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from progress.bar import IncrementalBar


#device = torch.device('cuda')
device = torch.device('cpu')


def make_train_data(file_name: str):
    with open(file_name) as file:
        train_lines = file.readlines()
        
    train_data = ' '.join(train_lines)
    train_data = list(re.sub(" +", " ", train_data))
    check_symbol = lambda c: c.isalpha() or c == '!' or c == '.' or c == '?' or c == ' ' or c == ',' or c == "'"
    train_data = [char for char in train_data if check_symbol(char)]
    return np.array(train_data)


def text_to_vec(train_data: np.array):
    chars, counts = np.unique(train_data, return_counts=True)

    dtype = [('char', 'U1'), ('count', int)]
    char_and_counts = np.array([(ch, -cnt) for ch, cnt in zip(chars, counts)], dtype=dtype)
    char_and_counts = np.sort(char_and_counts, order='count')
    sorted_chars = char_and_counts['char']

    char_to_indx = {char: indx for indx, char in enumerate(sorted_chars)}
    indx_to_char = {indx: char for indx, char in enumerate(sorted_chars)}
    vec = np.array([char_to_indx[word] for word in train_data])
    
    return vec, char_to_indx, indx_to_char


def get_batch(vec: np.array, lenght: int = 256, batch_size: int = 16):
    trains = []
    targets = []
    for _ in range(batch_size):
        start_pos = np.random.randint(0, len(vec) - lenght)
        chunk = vec[start_pos : start_pos + lenght]
        
        trains.append(torch.LongTensor(chunk[:-1]).view(-1, 1))
        targets.append(torch.LongTensor(chunk[1:]).view(-1, 1))
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)


class LSTM(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 embedding_size: int, 
                 num_of_layers: int = 1,
                 dropout_p: float = 0.2):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers

        self.encoder = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_of_layers)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(hidden_size, input_size)
        
    def forward(self, input, hidden):
        input_embd = self.encoder(input).squeeze(2)
        out, hidden = self.lstm(input_embd, hidden)
        out = self.dropout(out)
        out = self.linear(out)
        return out, hidden
    
    def init_hidden(self, batch_size: int = 1):
        return (torch.zeros(self.num_of_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
               torch.zeros(self.num_of_layers, batch_size, self.hidden_size, requires_grad=True).to(device))
        
    def train_model(self,
                    vec: np.array,
                    batch_size: int = 32,
                    num_of_epochs: int = 5000):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, amsgrad=True)

        bar = IncrementalBar('Countdown', max=num_of_epochs)

        for epoch in range(num_of_epochs):
            bar.next()
            self.train()
            train, target = get_batch(vec, batch_size=batch_size)
            train = train.permute(1, 0, 2).to(device)
            target = target.permute(1, 0, 2).to(device)
            hidden = self.init_hidden(batch_size)

            output, hidden = self(train, hidden)
            loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        

def evaluate(model: LSTM,
             char_to_indx: map,
             indx_to_char: map,
             start_text: str = ' ',
             prediction_len: int = 200,
             temp: float = 0.3):
    predicted_text = ''

    hidden = model.init_hidden(batch_size=1)
    input = [char_to_indx[char] for char in start_text]

    input = torch.LongTensor(input).view(-1, 1, 1).to(device)
    _, hidden = model(input, hidden)

    input = input[-1].view(-1, 1, 1)
    for i in range(prediction_len):
        output, hidden = model(input, hidden)
        output = output.cpu().data.view(-1)

        probability = F.softmax(output / temp, dim=-1).detach().cpu().data.numpy()        
        top_index = np.random.choice(len(char_to_indx), p=probability)
        predicted_text += indx_to_char[top_index]

        input = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)

    return start_text + predicted_text


def evaluate_word_type(model: LSTM,
                      word_to_indx: map,
                      indx_to_word: map,
                      start_text: str = ' ',
                      prediction_len: int = 200,
                      temp: float = 0.3):
    predicted_text = ''

    hidden = model.init_hidden(batch_size=1)
    input = [word_to_indx[word] for word in start_text.split(' ')]

    input = torch.LongTensor(input).view(-1, 1, 1).to(device)
    _, hidden = model(input, hidden)

    input = input[-1].view(-1, 1, 1)
    for i in range(prediction_len):
        output, hidden = model(input, hidden)
        output = output.cpu().data.view(-1)

        probability = F.softmax(output / temp, dim=-1).detach().cpu().data.numpy()        
        top_index = np.random.choice(len(word_to_indx), p=probability)
        predicted_text += indx_to_word[top_index] + ' '

        input = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)

    return start_text + predicted_text


def start_train(file_name):
    train_data = make_train_data(file_name)
    vec, char_to_indx, indx_to_char = text_to_vec(train_data)
    model = LSTM(input_size=len(indx_to_char), hidden_size=128, embedding_size=128, num_of_layers=2, dropout_p=0.4).to(device)
    model.train_model(vec)
    return model, char_to_indx, indx_to_char


def start_sampling(model, char_to_indx, indx_to_char, num_of_samples = 10):
    model.eval()

    for _ in range(num_of_samples):
        print(evaluate(
            model,
            char_to_indx,
            indx_to_char,
            temp=0.6,
            prediction_len=100,
            start_text='. '
            )
        )


def perplexity():
    s = ''.join(list(train_data))
    s = re.sub(r'[^\w\s]','', s)
    s = s.split()
    train_for_perp = s

    class perplexity:
        def __init__(self, train):
            train_ = [w.lower() for w in train]
            words, counts = np.unique(train_, return_counts=True)
            self.p = dict()
            n = len(train_)
            for w, c in zip(words, counts):
                self.p.update({w: c * 1. / n})

        def calc(self, test):
            test_ = [w.lower() for w in test]
            words, counts = np.unique(test_, return_counts=True)
            n = len(test)
            sum = 0
            for w, c in zip(words, counts):
                if w in self.p.keys():
                    sum += np.log(self.p[w]) * c
            return 1 / np.exp(sum * 1. / n)

    calc_perplixity = perplexity(train_for_perp)

    prediction_len = 300


    sum = 0
    for _ in range(100):
        model.eval()
        test = evaluate(
                model,
                char_to_indx,
                indx_to_char,
                temp=0.5,
                prediction_len=prediction_len,
                start_text='. '
                )
        s = ''.join(list(test))
        s = re.sub(r'[^\w\s]','', s)
        s = s.split()
        if len(s) > 0:
          perp = calc_perplixity.calc(s)
          sum += perp

    print(f'LSTM perplexity: {sum / 100}')

    true_text = train_for_perp
    perp = calc_perplixity.calc(true_text)
    print(f'original perplexity: {perp}')


from sys import argv


def main():
    input_file = "Ross.txt"
    if len(argv) > 1:
        input_file = argv[1]
    print('train...')
    args = start_train(input_file)
    print()
    start_sampling(*args, 10)


if __name__ == "__main__":
    main()
    