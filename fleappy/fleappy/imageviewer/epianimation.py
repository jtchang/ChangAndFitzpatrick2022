import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class EpiAnimation(animation.TimedAnimation):
    def __init__(self, responses, masks, vrange=(0, -1), interval=200):

        self.t = np.linspace(0, responses.shape[0])
        fig = plt.figure(figsize=[5, 5], dpi=300)
        self.ax = fig.add_subplot(111)
        self.responses = responses
        self.masks = masks
        self.range = vrange
        super(EpiAnimation, self).__init__(fig, interval=interval)

    def _draw_frame(self, framedata):
        self.ax.clear()

        self.ax.imshow(self.responses[framedata],
                       vmin=self.range[0],
                       vmax=self.range[1],
                       cmap='gray')

    def new_frame_seq(self):
        return iter(range(self.t.size))
