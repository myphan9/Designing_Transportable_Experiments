
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



file_name = "all_results/all_results.csv.gz"
df_merged = pd.read_csv(file_name)
df_merged = df_merged.loc[df_merged['metric']=='ATEError']
means = {}
stds = {}
se_of_msqs = {}
mean_squares = {}
sample_variances = {}
se_of_sample_variances = {}

for x in df_merged.x_value.unique():
    print(x)

    df = df_merged.loc[df_merged['x_value'] == x]
    means[x] = {}
    stds[x] = {}
    se_of_msqs[x] = {}
    mean_squares[x] = {}

    for d in df_merged.design.unique():

        df_x = df.loc[df['design']==d]
        means[x][d] = []
        stds[x][d] = []
        se_of_msqs[x][d] = []
        mean_squares[x][d] = []

        l = np.array(df_x['value'])
        l2 = l**2

        mean = l.mean()
        std = l.std()



        for t in range(1,1+len(l)):

            means[x][d].append(l[0:t].mean())
            stds[x][d].append(l[0:t].std())
            mean_squares[x][d].append(l2[0:t].mean())
            se_of_msqs[x][d].append(l2[0:t].std()/np.sqrt(t))

        mean_squares[x][d] = np.array(mean_squares[x][d])
        se_of_msqs[x][d] = np.array(se_of_msqs[x][d])
        plt.errorbar([d], [mean], [std], label = d, marker = 'o')
    plt.ylabel('ATEhat - ATE')
    plt.xticks([])
    plt.axhline(y=0, color = 'black')
    plt.legend(bbox_to_anchor=(1, 0))
    plt.savefig('figures/main_x' + str(x) +'.png',  bbox_inches='tight')

    plt.clf()
    for d in stds[x]:
        plt.plot(stds[x][d], label = d)
    plt.ylabel('STD')
    plt.xlabel('iteration')
    plt.legend(bbox_to_anchor=(1, -0.1))
    plt.savefig('figures/std_vs_iteration_x'+str(x)+'.png',  bbox_inches='tight')

    plt.clf()

    for d in mean_squares[x]:
        if True:
            c=np.random.rand(3,)

            plt.fill_between(range(len(mean_squares[x][d])), mean_squares[x][d] - se_of_msqs[x][d], mean_squares[x][d] + se_of_msqs[x][d],color = c, alpha=0.2)
            plt.plot(mean_squares[x][d], label = d, color = c)

    plt.ylabel('MSE')
    plt.xlabel('iteration')
    plt.legend(bbox_to_anchor=(1, -0.1))
    plt.savefig('figures/mse_vs_iteration_x' + str(x)+'.png',  bbox_inches='tight')
    plt.clf()
