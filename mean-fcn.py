import numpy as np
import seaborn as sns
fcn = np.array([
74.33,
87,
72.09,
68.71,
88.8,
98.8,
70.1,
95.89,
72.62,
79.73,
50.58,
64.38
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
57.03,
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
56.25,
])
fcner3 = np.array([
66.33,
93.5,
71.51,
67.77,
86.6,
98.27,
64.38,
95.5,
74.43,
78.9,
51.3,
56.25,

])
fcner5 = np.array([
66.67,
92,
69.28,
68.49,
89.7,
98.53,
66.38,
95.62,
73.93,
80.96,
50.39,
64.06,
])
wfcner33 = np.array([
63.67,
86.5,
71.8,
67.05,
89.8,
95.2,
67.24,
95.64,
77.21,
70.27,
45.58,
52.97,

])
wfcner55 = np.array([
65.33,
76.5,
73.02,
65.54,
87.5,
98,
64.29,
96.14,
76.07,
64.79,
44.48,
59.84,


])
wfcner35 = np.array([
65.33,
76.5,
73.02,
65.54,
87.5,
98,
64.29,
96.14,
76.07,
64.79,
44.48,
59.84,
])
col = []
for j in [fcn,wfcn3,wfcn5,fcner3,fcner5,wfcner33,wfcner55,wfcner35]:
    row = []
    for i in [fcn,wfcn3,wfcn5,fcner3,fcner5,wfcner33,wfcner55,wfcner35]:
        row.append(100*2/12*sum((j - i)/(j + i)))
    col.append(row)
labels = ["FCN", "WFCN3", "WFCN5", "FCNER3", "FCNER5", "WFCN33", "WFCN55", "WFCN35"]
sns.set(font_scale=0.6)
plot = sns.heatmap(col, annot=True, xticklabels =labels, yticklabels= labels)
plot.xaxis.set_label_position('top')
plot.xaxis.tick_top()
fig = plot.get_figure()
fig.savefig("fcn-mean.png")
# print(100*2/12*sum((wfcn3 - fcn)/(wfcn3 + fcn)))