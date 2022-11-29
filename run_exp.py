from exp_utils import Distribution, sample_and_train, grid_search

DEFAULTS = {
    'xdelta_pb': Distribution(mu=0., sigma=0.), # mu=0.001 if increasing returns is OFF
    'xdelta_ir': Distribution(mu=0.1, sigma=0.), # mu=0.1 if increasing returns is ON
    'xa_1': Distribution(mu=0.002, sigma=0.0007), # mu=0. if pref hetero is OFF
    'xa_2': Distribution(mu=0.008, sigma=0.01), # mu=0.00236 if pref hetero is OFF
    'xp_b': Distribution(mu=550, sigma=200), # sigma=0. if pref hetero is OFF
    'xb_0': Distribution(mu=0.3, sigma=0.15), # mu=0. if private goods is OFF
}

BEST_MODEL = {
    'xb_0': Distribution(mu=0.3, sigma=0.15),
    'xdelta_ir': Distribution(mu=0.16, sigma=0.),
    'xa_2': Distribution(mu=0.015, sigma=0.02),
    'xp_b': Distribution(mu=550, sigma=175),
}

VANILLA_RICEN = {
    'xdelta_pb': Distribution(mu=0.001, sigma=0.), # mu=0.001 if increasing returns is OFF
    'xdelta_ir': Distribution(mu=0., sigma=0.), # mu=0.1 if increasing returns is ON
    'xa_1': Distribution(mu=0., sigma=0.), # mu=0. if pref hetero is OFF
    'xa_2': Distribution(mu=0.00236, sigma=0.), # mu=0.00236 if pref hetero is OFF
    'xp_b': Distribution(mu=550, sigma=0), # sigma=0. if pref hetero is OFF
    'xb_0': Distribution(mu=0., sigma=0.), # mu=0. if private goods is OFF
}

if __name__ == '__main__':
    # Vanilla train
    # sample_and_train('vanilla', 1, DEFAULTS, VANILLA_RICEN)

    # Best model
    # sample_and_train('best', 1, DEFAULTS, BEST_MODEL)

    # Private goods
    # grid_search("b_0", DEFAULTS, "xb_0", "mu", 0.0, 0.0, amt=1)

    # Increasing returns
    grid_search("ir", DEFAULTS, "xdelta_ir", "mu", 0.324, 0.388, amt=3)

    # Damage heterogeneity
    # grid_search("damage", DEFAULTS, "xa_2", "sigma", 0.0672, 0.08, amt=3, overrides={"xa_2": Distribution(mu=0.015, sigma=0.01)})

    # Mitigation cost heterogeneity
    # grid_search("mitigation", DEFAULTS, "xp_b", "sigma", 0, 0, amt=1)

