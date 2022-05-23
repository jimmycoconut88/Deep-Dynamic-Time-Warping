import numpy as np
import seaborn as sns
wsr = np.array([
80.67,
71,
70.14,
65.18,
87,
91.73,
72.19,
96.74,
62.46,
44.52,
54.61,
])

mwsr = np.array([
86.67,
69.5,
65.4,
61.29,
86.3,
93,
69.52,
96.16,
60.33,
42.47,
53.31,
])

dwsr1 = np.array([
68.67,
75,
70.65,
64.53,
85.1,
90.53,
68.48,
96.08,
66.07,
52.05,
54.35,
])
dmwsr1 = np.array([
83,
73.5,
67.34,
63.02,
85,
89,
63.52,
96.32,
63.77,
54.25,
53.96,
])
dwsr2 = np.array([
78.33,
68.5,
71.44,
62.45,
90.2,
95.47,
68.57,
95.88,
71.97,
63.56,
54.48,
])

dmwsr2 = np.array([
86.67,
73,
70.36,
61.08,
89.8,
93.53,
68,
96.29,
70.66,
58.63,
54.48,
])

dwsr3 = np.array([
77,
63.5,
70,
64.82,
90.1,
95.13,
68.76,
95.88,
73.44,
61.1,
54.48,
])

dmwsr3 = np.array([
79,
67,
68.42,
61.94,
90.9,
92.67,
67.81,
95.76,
72.3,
62.47,
54.03,
])



col = []
for j in [wsr,mwsr, dwsr1, dmwsr1, dwsr2, dmwsr2, dwsr3, dmwsr3]:
    row = []
    for i in [wsr,mwsr, dwsr1, dmwsr1, dwsr2, dmwsr2, dwsr3, dmwsr3]:
        row.append(100*2/11*sum((j - i)/(j + i)))
    col.append(row)
labels = ["WSR", "MWSR", "DWSR1", "DMWSR1", "DWSR2", "DMWSR2", "DWSR3", "DMWSR3"]
sns.set(font_scale=0.6)
plot = sns.heatmap(col, annot=True, xticklabels =labels, yticklabels= labels)
plot.xaxis.tick_top()
plot.xaxis.set_label_position('top')
fig = plot.get_figure()
fig.savefig("dwsr-mean.png")