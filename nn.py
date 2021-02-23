from typing import Any

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os
import pll

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

input_size = 1
output_size = 1
learning_rate = 0.01
iterations = 1024 * 8

min_frequency = 45
max_frequency = 55
delta_frequency = max_frequency - min_frequency
duration = 1 / (2 * np.pi * max_frequency * 10)
ref_frequency = min_frequency + torch.rand(1) * delta_frequency
ref_frequency = ref_frequency * 2 * np.pi
phase = torch.rand(1) * 2 * np.pi


# train_loader = DataLoader(train_ds, batch_size, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size)
def accuracy(output, labels):
    diff = torch.sum(torch.abs(output - labels))
    return diff


# Neural network model
class NeuralPLLModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_dim = 32
        self.rnn_layers = 2

        self.rnn = nn.RNN(input_size,  # input size
                          self.hidden_dim,
                          self.rnn_layers)
        self.output_layer = nn.Linear(self.hidden_dim, output_size)

        self.pll = pll.PLL()
        self.data = torch.zeros(1)
        self.time = duration
        self.labels = self.get_sin_labels()

    def forward(self, xb):
        xb_rnn = xb.view(1, 1, input_size)
        batch_size = xb_rnn.size(0)
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(xb_rnn, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.output_layer(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.rnn_layers, batch_size, self.hidden_dim)
        return hidden

    def training_step(self):
        control_signal, _ = self(self.data)
        control_signal = control_signal.flatten()
        next_value = self.pll.forward_VCO(control_signal, duration)
        self.data = next_value
        self.time += duration
        self.labels = self.get_sin_labels()
        return self.loss_func(), self.pll.vco_freq.item(), next_value

    def step_end(self):
        self.data = self.data.detach()
        self.pll.detach()

    def get_sin_labels(self):
        start_from = self.time
        return generate_sin(start_from, self.time, duration)

    def loss_func(self):
        freq_diff = torch.abs(self.pll.vco_freq - ref_frequency)
        ref_phase = (self.time*ref_frequency + phase - self.pll.vco_phase) % (2*np.pi)
        phase_diff = torch.abs(self.pll.vco_phase - ref_phase)
        return freq_diff


def generate_sin(start, end, duration):
    time_series = torch.arange(start, end + duration / 2, duration)
    return torch.sin(time_series * ref_frequency + phase)


def fit(iterations,
        learning_rate,
        model,
        opt_func=torch.optim.RMSprop):
    """Train the model using gradient descent"""
    losses = []
    vco_freqs = []
    vco_outs = []
    optimizer = opt_func(model.parameters(), learning_rate)
    for it in range(iterations):
        percentile_count = round(iterations / 10) + 1
        if it % percentile_count == 0:
            print('{}%'.format(round(100 * it / iterations)))

        loss, vco_freq, vco_out = model.training_step()
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        model.step_end()

        # log data
        losses.append(loss.item())
        vco_freqs.append(vco_freq / 2 / np.pi)
        vco_outs.append(vco_out.item())

    return losses, vco_freqs, vco_outs


# generate data and data loaders

model = NeuralPLLModel()

fig, (ax1, ax2) = plt.subplots(2, 2)
fig.suptitle('NN-PLL training')

# Plot reference signal
ax1[0].set_title('Reference signal')
ax1[0].set_xlabel('time (s)')
reference_signal = generate_sin(0, duration * iterations, duration)
ax1[0].plot(reference_signal[-100:])

print('Train NN')
losses, vco_freqs, vco_outs = fit(iterations,
                                  learning_rate,
                                  model)

ax1[1].set_title('VCO vs Reference')
ax1[1].plot(reference_signal[-200:])
ax1[1].plot(vco_outs[-200:], color='r')

ax2[0].set_title('Training losses')
ax2[0].set_ylabel('Loss')
ax2[0].set_xlabel('iteration')
ax2[0].plot(losses[::10])

ax2[1].set_title('VCO frequency. Ref = {:.1f}Hz'.format(ref_frequency.item() / 2 / np.pi))
ax2[1].set_xlabel('iteration')
ax2[1].axline((0, ref_frequency.item() / 2 / np.pi), (1, ref_frequency.item() / 2 / np.pi))
ax2[1].plot(vco_freqs[::10], color='r')

plt.show()
