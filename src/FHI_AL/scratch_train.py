from FHI_AL.procedures import StandardMACEEnsembleProcedure
from yaml import safe_load


def main():
    with open("./mace_settings.yaml", "r") as file:
        mace_settings = safe_load(file)
    with open("./active_learning_settings.yaml", "r") as file:
        al_settings = safe_load(file)

    standard_ensemble = StandardMACEEnsembleProcedure(
        mace_settings=mace_settings,
        active_learning_settings=al_settings
    )

    standard_ensemble.train()
    standard_ensemble.test(
        path_to_ds="/home/users/u101418/al_aims/data/naphtalene/nve_295K_extxyz/out-1/out1.xyz"
    )
#

if __name__ == "__main__":
    main()