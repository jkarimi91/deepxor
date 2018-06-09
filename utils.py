from collections import defaultdict
from itertools import product
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def create_parity_data(seq_len=50, fixed=True):
    bits = [0, 1]
    num_examples = 100000

    inputs = []
    outputs = []
    for _ in xrange(num_examples):
        seq = [random.choice(bits) for _ in xrange(seq_len if fixed else random.randint(1, seq_len))]
        inputs.append(seq)
        outputs.append(sum(seq) % 2)
    return inputs, outputs

def pad_seqs(seqs):
    padded_seqs = []
    seq_lens = []
    pad_len = max(map(len, seqs))
    for i in xrange(len(seqs)):
        seq_lens.append(len(seqs[i]))
        padded_seqs.append(seqs[i] + [0] * (pad_len - len(seqs[i])))
    return padded_seqs, seq_lens

def create_xor_data(operator):
    bits = [0, 1]
    num_input_bits = 2

    inputs = [list(pair) for pair in product(bits, repeat=num_input_bits)]
    outputs = [operator(*pair) for pair in inputs]
    return inputs, outputs


def train(model, inputs, outputs, optimizer, num_iters, batch_size=None):
    losses = []
    loss_fxn = nn.BCELoss()

    for _ in xrange(num_iters):
        if batch_size is not None:
            idx = np.random.randint(len(inputs), size=batch_size)
            input_batch = inputs[idx]
            output_batch = outputs[idx]
        else:
            input_batch = inputs
            output_batch = outputs

        input_batch = Variable(input_batch, requires_grad=True)
        output_batch = Variable(output_batch)
        
        predictions = model(input_batch)
        loss = loss_fxn(predictions, output_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])
    return losses


def create_loss_plot(losses, title):
    plt.plot(losses);
    plt.xlabel('iters');
    plt.ylabel('loss');
    plt.title(title);


def create_decision_boundary_plot(model, inputs, outputs, title):
    # https://stackoverflow.com/questions/22294241/plotting-a-decision-boundary-separating-2-classes-using-matplotlibs-pyplot
    # https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels

    groups = defaultdict(list)
    for i in xrange(len(outputs)):
        groups[outputs[i]].append(inputs[i])

    colors = ['blue', 'green']
    for output, input in groups.iteritems():
        plt.scatter(*zip(*input), c=colors[output], label=output);

    plt.xlabel('input 1');
    plt.ylabel('input 2');
    plt.title(title);
    plt.legend(loc=1)

    bits = [0, 1]
    step = 0.01
    x_min, x_max = min(bits) - 1, max(bits) + 1
    y_min, y_max = x_min, x_max

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )

    mesh_inputs = np.c_[xx.ravel(), yy.ravel()]
    try:
        predictions = model(Variable(torch.Tensor(mesh_inputs), requires_grad=False))
    except:
        # add axis; time dependent model i.e. rnn
        predictions = model(Variable(torch.Tensor(mesh_inputs).unsqueeze(-1), requires_grad=False))

    predictions = predictions.data.numpy().reshape(xx.shape)
    plt.contour(xx, yy, np.around(predictions), colors=['red']);
    plt.plot();
