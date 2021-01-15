import torch
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def save_data_animation(filename, data, skip=-1):
    frame_rate = 30
    animation_duration = 3

    if skip == -1:
        total_size = len(data)
        number_of_frames = frame_rate * animation_duration
        skip = round(total_size / number_of_frames)

    to_save = data[::skip]
    fig = plt.figure()
    data_size = len(to_save[0])
    y_min = torch.min(to_save[0]).item()
    y_max = torch.max(to_save[0]).item()
    for element in to_save:
        y_min = min(torch.min(element).item(), y_min)
        y_max = max(y_max, torch.max(element).item())
    ax = plt.axes(xlim=(0, data_size), ylim=(y_min*1.1, y_max*1.1))
    scatter_plot = ax.scatter([], [], s=1)

    def init():
        print('Saving ' + filename)
        scatter_plot.set_offsets([])
        return scatter_plot,

    def animate(i):
        x = np.arange(data_size)
        y = to_save[i].numpy()
        offsets = np.stack((x, y)).T
        scatter_plot.set_offsets(offsets)
        percentile_count = round(len(to_save)/10) + 1
        if (i % percentile_count == 0):
            print('{}%'.format(round(100 * i / len(to_save))))
        return scatter_plot,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(to_save), interval=20, blit=False)
    writergif = animation.PillowWriter(fps=frame_rate)
    anim.save(filename, writer=writergif)
    plt.close(fig)
