import argparse

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

def get_seed():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", "-s", type=int, default=100, help-"Seed for reproducibility"
    )
    args = parser.parse_args()

    return args.seed

if __name__ == '__main__':
    seed = get_seed()
    print(f"Seeded with {seed}")

    # Vanilla train
    sample_and_train('vanilla', 1, DEFAULTS, VANILLA_RICEN, seed=seed)

    # Best model
    sample_and_train('best', 1, DEFAULTS, BEST_MODEL, seed=seed)

    # Private goods
    grid_search("b_0", DEFAULTS, "xb_0", "mu", 0.0, 0.6, amt=12, seed=seed)

    # Increasing returns
    grid_search("ir", DEFAULTS, "xdelta_ir", "mu", 0.0, 0.4, amt=12, seed=seed)

    # Damage heterogeneity
    grid_search("damage", DEFAULTS, "xa_2", "sigma", 0.0, 0.08, amt=12, overrides={"xa_2": Distribution(mu=0.015, sigma=0.01)}, seed=seed)

    # Mitigation cost heterogeneity
    grid_search("mitigation", DEFAULTS, "xp_b", "sigma", 0, 500, amt=12, seed=seed)

