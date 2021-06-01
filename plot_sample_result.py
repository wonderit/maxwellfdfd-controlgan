import pandas as pd

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
font_size = 20
plt.rcParams.update({'font.size': font_size})
# csv_path = './figures/sample/1000nm_controlgan_9/1000_9_real_controlgan.csv'
csv_path = './figures/sample/1250nm_controlgan_2/1250_2_real_controlgan.csv'
# csv_path = './figures/sample/1500nm_controlgan_2/1500_2_real_controlgan.csv'

x_axis = range(400, 1600, 50)
df = pd.read_csv(csv_path, sep=',', header=None, names=x_axis)
print(df.values)

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.plot(x_axis, df.values[0], label='Real', color='black')
myeongjo = 'NanumMyeongjo'
ax.plot(x_axis, df.values[1], label ='Predict', color = 'blue')
#
# peaks_positive, _ = find_peaks(y_test, height=0)
# peaks_negative, _ = find_peaks(1 - y_test, height=0)
# mask = np.zeros_like(y_test, np.bool)
# mask[peaks_positive] = 1
# mask[peaks_negative] = 1
# peak_array = mask * y_test
# peak_array[peak_array == 0] = np.nan
# plt.plot(x_axis, peak_array, "o", markersize=10)


ax.set_title(r'predict simulation', fontsize=font_size, fontname = myeongjo)
ax.set_xlabel('wavelength', fontsize=font_size, fontname = myeongjo)
ax.set_ylabel('transmittance', fontsize=font_size, fontname = myeongjo)
ax.legend(loc='upper left', fontsize=font_size)
# ax.legend(loc = 'lower center', fontsize = 14)
ax.grid(True)
ax.set_xlim(400, 1550)
ax.set_ylim(0, 1)


# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_major_locator(MultipleLocator(0.1))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(50))
ax.grid(which='major', color='#CCCCCC', linestyle='--')
ax.grid(which='minor', color='#CCCCCC', linestyle=':')

# ax.set_ylim(0, 10000)
# ax.set_yticks(np.arange(0, 10000 + 1, 2500))

fig.tight_layout()
fig.set_size_inches(11, 8)
# plt.savefig('./figures/sample/1000nm_controlgan_9/controlgan_simulation_font{}.png'.format(font_size))
plt.savefig('./figures/sample/1250nm_controlgan_2/controlgan_simulation_font{}.png'.format(font_size))
# plt.savefig('./figures/sample/1500nm_controlgan_2/controlgan_simulation_font{}.png'.format(font_size))
plt.show()


