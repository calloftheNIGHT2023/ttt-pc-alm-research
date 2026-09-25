"""387 predeclared 36 methods, two independently executed budgets, fresh pilot."""
SEEDS = list(range(6700000, 6700032))
BUDGETS = [.5, 1.]
PRIMARY = 'regional_active1024_bfs_observed'
PRIMARY_BUDGET = .5
BLOCK_SIZE = 4


def configs():
    answer = []
    for method in ['regional_active512', 'regional_active1024', 'regional_passive1024', 'pdhg_cold1024', 'pdhg_box1024']:
        for schedule, ordering in [('bfs', 'observed'), ('dfs', 'farthest_x')]:
            answer.append(dict(name=f'{method}_{schedule}_{ordering}', kind='regional', method=method,
                schedule=schedule, ordering=ordering))
    for method in ['linear_ls', 'residual_linear_ls', 'residual_linear_ridge', 'residual_linear_rls',
                   'prior4096_ridge', 'prior16384_ridge', 'prior65536_ridge', 'rbf_loocv']:
        answer.append(dict(name=method, kind='regression', method=method))
    for method in ['meta_ridge64', 'meta_ridge128', 'meta_shallow64_5', 'meta_shallow64_20']:
        answer.append(dict(name=method, kind='meta', method=method, warm_trajectory='shallow' in method))
    for family, steps, restarts in [('adam', 240, 64), ('gauss_newton', 40, 64), ('pc', 128, 64),
            ('alm', 128, 64), ('nodual', 128, 64), ('adam', 240, 256), ('gauss_newton', 40, 256)]:
        for readout in ['point', 'posterior_union']:
            answer.append(dict(name=f'{family}{steps}_r{restarts}_{readout}', kind='optimizer', family=family,
                steps=steps, restarts=restarts, readout=readout))
    assert len(answer) == len({c['name'] for c in answer}) == 36
    return answer


MAIN_CONTROLS = ['prior65536_ridge', 'meta_ridge128', 'meta_shallow64_20',
    'adam240_r256_posterior_union', 'gauss_newton40_r256_posterior_union',
    'pdhg_cold1024_bfs_observed', 'pdhg_box1024_dfs_farthest_x', 'regional_passive1024_bfs_observed']
