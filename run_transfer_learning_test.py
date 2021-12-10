import pandas as pd
from tqdm import tqdm

import dgp
import design
from plan import Plan
import estimator as est
import evaluator as evalr
import numpy as np
from scipy.stats import multivariate_normal, uniform

def make_plan(designs, X_source_dist, X_target_dist):
    plan = Plan()

    for d in designs:
        name = d[0]
        dgn = d[1]
        estr = d[2]
        design_kwargs = d[3]
        design_kwargs['source'] = X_source_dist
        design_kwargs['target'] = X_target_dist
        plan.add_design(name, dgn, estr, design_kwargs)

    plan.add_evaluator('ATEError', evalr.ATEError)

    return plan

NUM_ITERS =  100
sample_size =  100 #np.arange(500, 2000, 200) #[500, 600, 8000, 1000, 2000]
num_covariates_list = [2] #np.arange(2, 6, 2) #10
weight_threshold = 2000


dfs = []

#sample_sizes = [10, 20, 32, 50, 64, 72, 86, 100]
dgp_factory_class_list = [dgp.SquareFactory]
TOTAL_PREV_ITERS = 0
for num_covariates in num_covariates_list:
    distance = 0.3
    X_source_dist = multivariate_normal(1 * np.ones(num_covariates),  1* np.eye(num_covariates))
    X_target_dist = multivariate_normal((1+distance) * np.ones(num_covariates), 1* np.eye(num_covariates))


    plan = make_plan([
    ('Rerandomization-No-Balance', design.ReRandomization, est.DifferenceInMeans, {'pr_accept':1, 'weight_threshold':weight_threshold}),
    ('Rerandomization-Balance-Source', design.ReRandomization, est.DifferenceInMeans, {'pr_accept':0.01, 'balance_on_target':False, 'weight_threshold':weight_threshold}),
    ('Rerandomization-Balance-Target', design.ReRandomization, est.DifferenceInMeans, {'pr_accept':0.01, 'balance_on_target':True, 'weight_threshold':weight_threshold}),
], X_source_dist, X_target_dist)

    print(f"Sample Size: {sample_size}")
    dgp_factory_list = [factory(N=sample_size) for factory in dgp_factory_class_list]
    for dgp_factory in dgp_factory_list:
        dgp_name = type(dgp_factory.create_dgp()).__name__
        print(f"DGP name: {dgp_name}")
        for it in tqdm(range(TOTAL_PREV_ITERS, TOTAL_PREV_ITERS + NUM_ITERS)):
            plan.add_env(dgp_factory,  seed=it * 1001, X_source_dist =X_source_dist,  X_target_dist =X_target_dist)

            for weighted_estimator in [True, False]:
                plan.use_weighted_estimator(weighted_estimator = weighted_estimator)

                for design_name in plan.designs:
                    result = plan.execute(design_name, weight_threshold = weight_threshold, weighted_estimator = weighted_estimator )
                    result['iteration'] = it
                    result['sample_size'] = sample_size
                    result['dgp'] = dgp_name
                    result['x_label'] = "Number of Covariates"
                    result['x_value'] = num_covariates

                    filename = f"results/{dgp_name}_{design_name}_weighted-estimator{weighted_estimator}_n{sample_size}_i{it}.csv.gz"
                    result.to_csv(filename, index=False)
                    dfs.append(result)



results = pd.concat(dfs)

filename = f"all_results/all_results.csv.gz"

# print(f"""
# \n**********************************************************************
# ***\tSAVING TO `{filename}`\t\t   ***
# **********************************************************************""")

results.to_csv(filename, index=False)
