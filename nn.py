import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
import gif_util as gif
import torch.nn as nn
import numpy as np
import os
import pll

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

input_size = 8  # ref + vco values history
output_size = 1
learning_rate = 0.01
iterations = 1024*4

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
    diff = torch.abs(output - labels)
    return diff


# Neural network model
class NeuralPLLModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(OrderedDict([
            ('input-1', nn.Linear(input_size, 16)),
            ('activation1', nn.ReLU()),
            ('1-2', nn.Linear(16, 8)),
            ('activation2', nn.ReLU()),
            ('2-3', nn.Linear(8, 4)),
            ('activation3', nn.ReLU()),
            ('3-output', nn.Linear(4, output_size))
        ]))
        self.pll = pll.PLL()
        self.data = torch.zeros(input_size)
        self.time = input_size * duration
        self.labels = self.get_sin_labels()

    def forward(self, xb):
        out = self.network(xb)
        return out

    def training_step(self):
        control_signal = self(self.data)
        next_value = self.pll.forward_VCO(control_signal, duration)
        self.data = torch.roll(self.data, -1, 0)
        self.data[input_size - 1] = next_value
        self.time += duration

        self.labels = self.get_sin_labels()
        return self.loss_fn(self.data, self.labels), self.pll.vco_freq.item(), next_value

    def step_end(self):
        self.data = self.data.detach()
        self.pll.detach()

    def get_sin_labels(self):
        start_from = self.time - (input_size - 1) * duration
        return generate_sin(start_from, self.time, duration)

    def loss_fn(self, outputs, labels):
        loss = (outputs[-1] - labels[-1])*(outputs[-1] - labels[-1])
        return loss.exp()


def generate_sin(start, end, duration):
    time_series = torch.arange(start, end + duration / 2, duration)
    return torch.sin(time_series * ref_frequency + phase)


def fit(iterations,
        learning_rate,
        model,
        opt_func=torch.optim.SGD):
    """Train the model using gradient descent"""
    losses = []
    weights = []
    grads = []
    vco_freqs = []
    vco_outs = []
    optimizer = opt_func(model.parameters(), learning_rate)
    weights.append(get_weights(model))
    for it in range(iterations):
        percentile_count = round(iterations/10) + 1
        if it % percentile_count == 0:
            print('{}%'.format(round(100 * it / iterations)))

        loss, vco_freq, vco_out = model.training_step()
        loss.backward()
        optimizer.step()

        #log gradiends
        grads.append(get_grads(model))

        optimizer.zero_grad()
        model.step_end()

        # log data
        losses.append(loss.item())
        weights.append(get_weights(model))
        vco_freqs.append(vco_freq/2/np.pi)
        vco_outs.append(vco_out.item())


    return losses, weights, grads, vco_freqs, vco_outs


def get_weights(model):
    w1 = model.network[0].weight.detach().clone()[0, :].flatten()
    w2 = model.network[0].weight.detach().clone()[1, :].flatten()
    w = torch.cat((w1, w2), 0)
    return w


def get_grads(model):
    w1 = model.network[0].weight.grad[0, :].flatten()
    w2 = model.network[0].weight.grad[1, :].flatten()
    w = torch.cat((w1, w2), 0)
    return w
# generate data and data loaders

model = NeuralPLLModel()

fig, (ax1, ax2, ax3) = plt.subplots(3, 2)
fig.suptitle('NN-PLL training')

# Plot reference signal
ax1[0].set_title('Reference signal')
ax1[0].set_xlabel('time (s)')
reference_signal = generate_sin(0, duration * iterations, duration)
ax1[0].plot(reference_signal[-100:])

print('Train NN')
losses, weights, grads, vco_freqs, vco_outs = fit(iterations,
                                         learning_rate,
                                         model)

gif.save_data_animation('weights.gif', weights)
gif.save_data_animation('grads.gif', grads)

ax1[1].set_title('VCO vs Reference')
ax1[1].plot(reference_signal[-100:])
ax1[1].plot(vco_outs[-100:], color='r')

ax2[0].set_title('Training losses')
ax2[0].set_ylabel('Loss')
ax2[0].set_xlabel('iteration')
ax2[0].plot(losses[::10])

ax2[1].set_title('VCO frequency. Ref = {:.1f}Hz'.format(ref_frequency.item()/2/np.pi))
ax2[1].set_xlabel('iteration')
ax2[1].axline((0, ref_frequency.item()/2/np.pi), (1, ref_frequency.item()/2/np.pi))
ax2[1].plot(vco_freqs[::10], color='r')

ax3[0].set_title('Gradients for weights layer 1')
ax3[0].scatter(np.arange(len(grads[0])), grads[0])

ax3[1].set_title('Gradients for weights layer 1')
ax3[1].scatter(np.arange(len(grads[100])), grads[100])
plt.tight_layout()

plt.show()
