from FHI_AL.tools.utilities import test_ensemble, ensemble_from_folder
from ase.io import read
import argparse
import numpy as np
from prettytable import PrettyTable
import logging

def main():

    # setup logger to save to 'test_ensemble.log'
    logging.basicConfig(filename='test_ensemble.log', level=logging.INFO)
    logging.info('Starting test.')
    parser = argparse.ArgumentParser(description='Test ensemble of models')
    parser.add_argument(
        '--models',
        type=str,
        help='Path to models',
        required=True
        )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to dataset',
        required=True
        )
    parser.add_argument(
        '--output_args',
        type=str,
        nargs='+',
        help='List of output arguments',
        default=['energy', 'forces']
        )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size',
        default=16
        )
    parser.add_argument(
        '--device',
        type=str,
        help='Device',
        default='cpu'
        )
    parser.add_argument(
        '--save',
        type=str,
        help='Path to folder where save results',
        default="./"
        )
    parser.add_argument(
        '--return_predictions',
        type=bool,
        help='Return predictions',
        default=False
    )
    
    parser.add_argument(
        '--energy_key',
        type=str,
        help='Energy key',
        default='REF_energy'
    )
    parser.add_argument(
        '--forces_key',
        type=str,
        help='Forces key',
        default='REF_forces'
    )
    parser.add_argument(
        '--stress_key',
        type=str,
        help='Stress key',
        default='REF_stress'
    )
    parser.add_argument(
        '--virials_key',
        type=str,
        help='Virials key',
        default='virials'
    )
    parser.add_argument(
        '--dipole_key',
        type=str,
        help='Dipole key',
        default='dipoles'
    )
    parser.add_argument(
        '--charges_key',
        type=str,
        help='Charges key',
        default='charges'
    )
    parser.add_argument(
        '--head_key',
        type=str,
        help='Head key',
        default='head'
    )
    

    args = parser.parse_args()

    ensemble = ensemble_from_folder(
        path_to_models=args.models,
        device=args.device
        )
    atoms_list = read(args.data, index=':')
    
    possible_args = ['energy', 'forces', 'stress', 'virials']
    output_args = {}
    for arg in args.output_args:
        arg = arg.lower()
        if arg not in possible_args:
            raise ValueError('Invalid output argument')
        if arg == 'energy':
            output_args['energy'] = True
        if arg == 'forces':
            output_args['forces'] = True
        if arg == 'stress':
            output_args['stress'] = True
        if arg == 'virials':
            output_args['virials'] = True
    
    for arg in possible_args:
        if arg not in output_args:
            output_args[arg] = False

    (avg_ensemble_metrics, ensemble_metrics) = test_ensemble(
        ensemble = ensemble,
        atoms_list = atoms_list,
        batch_size = args.batch_size,
        output_args = output_args,
        device = args.device,
        return_predictions = args.return_predictions,
        energy_key = args.energy_key,
        forces_key = args.forces_key,
        stress_key = args.stress_key,
        virials_key = args.virials_key,
        dipole_key = args.dipole_key,
        charges_key = args.charges_key,
        head_key = args.head_key
    )
    results = {
        'avg_ensemble_metrics': avg_ensemble_metrics,
        'ensemble_metrics': ensemble_metrics
    }
    np.savez(args.save + 'ensemble_test_results.npz', **results)


    table = PrettyTable()
    table.field_names = ['Model', 'MAE E [meV/Atom]', 'RMSE E [meV/Atom]', 'MAE F [meV/(A*atom)]', 'RMSE F [meV/(A*atom)]']
    for i in ensemble.keys():
        table.add_row(
            [
                i,
                f"{ensemble_metrics[i]['mae_e_per_atom'] * 10e2:.1f}",
                f"{ensemble_metrics[i]['rmse_e_per_atom'] * 10e2:.1f}",
                f"{ensemble_metrics[i]['mae_f'] * 10e2:.1f}",
                f"{ensemble_metrics[i]['rmse_f'] * 10e2:.1f}"
            ]
        )
    table.add_row(
        [
            'Average',
            f"{avg_ensemble_metrics['mae_e_per_atom'] * 10e2:.1f}",
            f"{avg_ensemble_metrics['rmse_e_per_atom'] * 10e2:.1f}",
            f"{avg_ensemble_metrics['mae_f'] * 10e2:.1f}",
            f"{avg_ensemble_metrics['rmse_f'] * 10e2:.1f}"
        ]
    )


    print(table)

if __name__ == '__main__':
    main()
