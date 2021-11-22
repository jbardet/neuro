import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from test_global_times import test_times
#lengths, times_mp, times_hol, times_h = test_times()
lengths, times_mp, times_h = test_times()

#times_mp = pd.read_csv('/home/james/neuro/Data/times_mp.csv')
#times_h = pd.read_csv('/home/james/neuro/Data/times_h.csv')
#times_hol = pd.read_csv('/home/james/neuro/Data/times_hol.csv')
#lengths = [31.6, 39.4, 31.6, 26.4, 37.8]

labels = times_mp.columns.tolist()
mp_means = times_mp[labels].mean().values
mp_std = times_mp[labels].std().values
#hol_means = times_hol[labels].mean().values
#hol_std = times_hol[labels].std().values
h_means = times_h[labels].mean().values
h_std = times_h[labels].std().values

x = np.arange(len(labels))  # the label locations
width = 0.8  # the width of the bars

fig, ax = plt.subplots()
#rects1 = ax.bar(x - width/2, lengths, width/4, label='Video Original Length')
#rects2 = ax.bar(x - width/4, mp_means, width/4, label='MediaPipe', yerr = mp_std)
#rects3 = ax.bar(x, hol_means, width/4, label='Holistic', yerr = hol_std)
#rects4 = ax.bar(x + width/4, h_means, width/4, label='SSD-MobileNet V2 model', yerr = h_std)

rects1 = ax.bar(x - width/3, lengths, width/3, label='Video Original Length')
rects2 = ax.bar(x, mp_means, width/3, label='MediaPipe', yerr = mp_std)
rects3 = ax.bar(x + width/3, h_means, width/3, label='SSD-MobileNet V2 model', yerr = h_std)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Processing time [s]')
ax.set_title('Duration by video in seconds')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)
#ax.bar_label(rects4, padding=3)

fig.tight_layout()
plt.savefig('/home/james/neuro/Data/plot_10.png')
plt.show()
plt.savefig('/home/james/neuro/Data/plot_10.png')
