import numpy as np
import matplotlib.pyplot as plt

# fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
#
# labels = ["Bass",
#           "Bariton",
#           "Tenor",
#           "Sopran"]
#
# fach = [0, 1, 0, 1, 0, 1, 0, 1]
#
# size = 0.3
#
# vals = np.array([[157, 60], [184, 157], [190, 199], [404, 348]])
#
# cmap = plt.get_cmap("tab20c")
# outer_colors = cmap(np.arange(4)*4)
# inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10, 13, 14]))
#
# wedges, texts = ax.pie(vals.sum(axis=1), labels=labels, radius=1, colors=outer_colors,
#        wedgeprops=dict(width=size, edgecolor='w'))
#
# ax.pie(vals.flatten(), labels=fach, labeldistance=0.75, radius=1-size, colors=inner_colors,
#        wedgeprops=dict(width=size, edgecolor='w'))
#
# ax.set(aspect="equal", title='Dateiverteilung')
#
# text = ['Labels:', '0: dramatisch', '1: lyrisch']
#
# plt.text(1.1, -0.7, text[1], fontsize=12)
# plt.text(1.1, -0.9, text[2], fontsize=12)
#
# plt.savefig('/Users/wzehui/Documents/MA/Plot/Dateiverteilung.pdf')
# plt.show()

# # pie
# fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
#
# labels = 'Dramatisch', 'Lyrisch'
# sizes = [157+184+190+404, 60+157+199+348]
#
# explode = (0.05, 0)
#
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, labels=labels, labeldistance=0.5, pctdistance=0.25, explode=explode, autopct='%1.1f%%', startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# ax1.set_title("Stimmfachverteilung")
#
# plt.savefig('/Users/wzehui/Documents/MA/Plot/Stimmfachverteilung.pdf')
#
# plt.show()

import numpy as np
import wave
import sys


spf = wave.open('/Users/wzehui/Documents/MA/Daten/quellcode/sounddb/B1-Amfortas/B1-m307-Amfo-A-c1-1_m.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')


#If Stereo
# if spf.getnchannels() == 2:
#     print('Just mono files')
#     sys.exit(0)

plt.figure(1)
plt.title('Signal Wave')
plt.plot(signal)
plt.savefig('/Users/wzehui/Documents/MA/Plot/wav_B1-m307-Amfo-A-c1-1_demo.pdf')
plt.show()