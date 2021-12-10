import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



file_name = "all_results/all_results.csv.gz"
df_merged = pd.read_csv(file_name)
df_merged = df_merged.loc[df_merged['metric']=='ATEError']


final_means = {}
final_mean_squares = {}
final_mean_squares_se = {}

final_var = {}
final_var_se = {}

final_bias = {}
final_bias_se = {}

for d in df_merged.design.unique():
    final_means[d] = {}
    final_mean_squares[d] = {}
    final_mean_squares_se[d] = {}

    final_var[d] = {}
    final_var_se[d] = {}

    final_bias[d] = {}
    final_bias_se[d] = {}
    df = df_merged.loc[df_merged['design'] == d]
    for x in df_merged.x_value.unique():
        df_x = df.loc[df['x_value']==x]


        l = np.array(df_x['value'])
        mean = l.mean()
        l2 = l**2

        var_list = (l-mean)**2
        var = var_list.mean()
        var_se = var_list.std()/np.sqrt(len(var_list))

        bias_list = l
        bias = bias_list.mean()
        bias_se = bias_list.std()/np.sqrt(len(bias_list))

        final_means[d][x] = l.mean()

        final_mean_squares[d][x] = l2.mean()
        final_mean_squares_se[d][x] = l2.std()/np.sqrt(len(l2))

        final_var[d][x] = var
        final_var_se[d][x] = var_se

        final_bias[d][x] = bias
        final_bias_se[d][x] = bias_se

for d in final_mean_squares:
    # https://stackoverflow.com/questions/37266341/plotting-a-python-dict-in-order-of-key-values
    msq_lists = sorted(final_mean_squares[d].items()) # sorted by key, return a list of tuples
    se_lists = sorted(final_mean_squares_se[d].items())
    x, y = zip(*msq_lists) # unpack a list of pairs into two tuples


    x, delta = zip(*se_lists)

    y = np.array(y)
    delta = np.array(delta)
    c=np.random.rand(3,)

    plt.fill_between(x, y - delta, y + delta,color = c, alpha=0.2)
    plt.plot(x, y, label = d, color = c)

plt.ylabel('MSE')
plt.xlabel('Distance')
plt.legend(bbox_to_anchor=(1, -0.1))
plt.savefig('figures/mse_vs_distance.png',  bbox_inches='tight')
plt.clf()
for d in final_var:
    # https://stackoverflow.com/questions/37266341/plotting-a-python-dict-in-order-of-key-values
    msq_lists = sorted(final_var[d].items()) # sorted by key, return a list of tuples
    se_lists = sorted(final_var_se[d].items())
    x, y = zip(*msq_lists) # unpack a list of pairs into two tuples


    x, delta = zip(*se_lists)

    y = np.array(y)
    delta = np.array(delta)
    c=np.random.rand(3,)

    plt.fill_between(x, y - delta, y + delta,color = c, alpha=0.2)
    plt.plot(x, y, label = d, color = c)

plt.ylabel('Variance')
plt.xlabel('Distance')
plt.legend(bbox_to_anchor=(1, -0.1))
plt.savefig('figures/var_vs_distance.png',  bbox_inches='tight')
plt.clf()
for d in final_bias:
    # https://stackoverflow.com/questions/37266341/plotting-a-python-dict-in-order-of-key-values
    msq_lists = sorted(final_bias[d].items()) # sorted by key, return a list of tuples
    se_lists = sorted(final_bias_se[d].items())
    x, y = zip(*msq_lists) # unpack a list of pairs into two tuples


    x, delta = zip(*se_lists)

    y = np.array(y)
    delta = np.array(delta)
    c=np.random.rand(3,)

    plt.fill_between(x, y - delta, y + delta,color = c, alpha=0.2)
    plt.plot(x, y, label = d, color = c)

plt.ylabel('Bias')
plt.xlabel('Distance')
plt.legend(bbox_to_anchor=(1, -0.1))
plt.savefig('figures/bias_vs_distance.png',  bbox_inches='tight')
plt.clf()
