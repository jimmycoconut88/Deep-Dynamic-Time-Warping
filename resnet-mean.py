import numpy as np
import seaborn as sns
ResNet1D = np.array([
75.67,
87,
99.64,
72.3,
64.89,
83.2,
74,
97.6,
95.96,
72.79,
81.64,
47.08,

])

wrn3 = np.array([
65.33,
86,
99.64,
72.73,
66.62,
86.8,
74.19,
99.2,
95.99,
73.61,
67.81,
50.52,

])

wrn5 = np.array([
71,
87.5,
99.64,
71.94,
66.26,
88.2,
74.1,
97.73,
95.88,
73.93,
75.75,
48.38,

])
rner3 = np.array([
69,
93.5,
99.64,
71.51,
66.55,
87.8,
73.62,
98.47,
95.86,
79.84,
81.64,
49.81,


])
rner5 = np.array([
64.33,
93,
99.64,
73.88,
66.76,
89.9,
71.81,
98.4,
96.08,
79.67,
80,
51.36,
])
wrner33 = np.array([
70.67,
81.5,
97.86,
72.3,
62.3,
87.6,
67.52,
98.33,
95.97,
82.13,
75.48,
48.31,
])
wrner55 = np.array([
66.33,
79.50,
100.00,
72.37,
66.26,
89.00,
72.19,
98.20,
96.05,
80.98,
70.00,
49.42,
])

col = []
for j in [ResNet1D,wrn3,wrn5,rner3,rner5,wrner33,wrner55]:
    row = []
    for i in [ResNet1D,wrn3,wrn5,rner3,rner5,wrner33, wrner55]:
        row.append(100*2/12*sum((j - i)/(j + i)))
    col.append(row)
labels = ["ResNet", "WRN3", "WRN5", "RNWR3", "RNWR5", "WRNWR33", "WRNWR55"]
sns.set(font_scale=0.6)
plot = sns.heatmap(col, annot=True, xticklabels =labels, yticklabels= labels)
plot.xaxis.tick_top()
plot.xaxis.set_label_position('top')
fig = plot.get_figure()
fig.savefig("resnet-mean.png")