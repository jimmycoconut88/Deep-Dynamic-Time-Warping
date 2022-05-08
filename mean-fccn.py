import numpy as np
import seaborn as sns
fcn = np.array([
72.67,
87,
67.63,
68.20,
88.40,
97.67,
68.87,
95.65,
72.95,
79.04,
50.39,
])
fccn = np.array([
69,
90.50,
69.71,
64.60,
86.10,
96.67,
62.67,
96.27,
71.48,
68.63,
47.53
])

wfcn3 = np.array([
70,
76,
73.02,
64.46,
87.9,
93.67,
68.29,
95.88,
71.31,
66.99,
49.74,
])

wfcn5 = np.array([
71,
80,
71.37,
67.63,
83.5,
98.73,
64,
95.84,
73.61,
68.9,
48.64,
])

col = []
for j in [fcn, fccn,wfcn3,wfcn5,]:
    row = []
    for i in [fcn, fccn, wfcn3,wfcn5,]:
        row.append(100*2/11*sum((j - i)/(j + i)))
    col.append(row)
labels = ["FCN", "FCCN", "WFCN3", "WFCN5"]
sns.set(font_scale=0.6)
plot = sns.heatmap(col, annot=True, xticklabels =labels, yticklabels= labels)
plot.xaxis.set_label_position('top')
plot.xaxis.tick_top()
fig = plot.get_figure()
fig.savefig("fccn-mean.png")