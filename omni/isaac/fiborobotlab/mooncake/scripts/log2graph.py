import matplotlib.pyplot as plt
import pandas as pd

plt.xlim([0, 500000])
plt.ylim([0, 6000])
fig=plt.figure()
d = {}
for i, filename in enumerate(['log_11111', 'log_01111', 'log_10111', 'log_11011', 'log_11101', 'log_11110']):
# path = "F:/mooncake_policy_01111/log.txt"
    f = open('log/'+filename+'.txt')
    lines = f.readlines()
    topic = lines[0][:-1]
    lines = lines[1:]

    ax=fig.add_subplot(6, 1, i+1)
    # ax = plt.subplot(6, 1, i+1)
    # ax.set_xlim(xmin=0.0, xmax=500000)
    ax.set_ylim(ymin=0.0, ymax=600)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    X, Y = [], []
    for line in lines:
        [x, y] = line[:-1].split('\t')
        X.append(x)
        Y.append(y)
    ax.plot(X, Y)
    d[topic+'x'] = X.copy()
    d[topic+'y'] = Y.copy()
plt.show()
df = pd.DataFrame(data=d)
df.to_csv('log.csv')