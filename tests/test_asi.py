import re
from ase.io import read
from ase.build import molecule
from ase.calculators.aims import Aims
import mpi4py.MPI as MPI
from asi4py import ASI_ASE_calculator
from ase.md.langevin import Langevin
from ase import units
import numpy as np
import ase
from time import sleep

string_patterns = {
    'xc': re.compile(r'^\s*(xc)\s+(\S+)', re.IGNORECASE),
    'spin': re.compile(r'^\s*(spin)\s+(\S+)', re.IGNORECASE),
    'communication_type': re.compile(r'^\s*(communication_type)\s+(\S+)', re.IGNORECASE),
    'density_update_method': re.compile(r'^\s*(density_update_method)\s+(\S+)', re.IGNORECASE),
    'KS_method': re.compile(r'^\s*(KS_method)\s+(\S+)', re.IGNORECASE),
    'mixer': re.compile(r'^\s*(mixer)\s+(\S+)', re.IGNORECASE),
    'output_level': re.compile(r'^\s*(output_level)\s+(\S+)', re.IGNORECASE),
    'packed_matrix_format': re.compile(r'^\s*(packed_matrix_format)\s+(\S+)', re.IGNORECASE),
    'relax_unit_cell': re.compile(r'^\s*(relax_unit_cell)\s+(\S+)', re.IGNORECASE),
    'restart': re.compile(r'^\s*(restart)\s+(\S+)', re.IGNORECASE),
    'restart_read_only': re.compile(r'^\s*(restart_read_only)\s+(\S+)', re.IGNORECASE),
    'restart_write_only': re.compile(r'^\s*(restart_write_only)\s+(\S+)', re.IGNORECASE),
    'total_energy_method': re.compile(r'^\s*(total_energy_method)\s+(\S+)', re.IGNORECASE),
    'qpe_calc': re.compile(r'^\s*(qpe_calc)\s+(\S+)', re.IGNORECASE),
    'species_dir': re.compile(r'^\s*(species_dir)\s+(\S+)', re.IGNORECASE),
    'run_command': re.compile(r'^\s*(run_command)\s+(\S+)', re.IGNORECASE),
    'plus_u': re.compile(r'^\s*(plus_u)\s+(\S+)', re.IGNORECASE),
}

bool_patterns = {
    'collect_eigenvectors': re.compile(r'^\s*(collect_eigenvectors)\s+(\S+)', re.IGNORECASE),
    'compute_forces': re.compile(r'^\s*(compute_forces)\s+(\S+)', re.IGNORECASE),
    'compute_kinetic': re.compile(r'^\s*(compute_kinetic)\s+(\S+)', re.IGNORECASE),
    'compute_numerical_stress': re.compile(r'^\s*(compute_numerical_stress)\s+(\S+)', re.IGNORECASE),
    'compute_analytical_stress': re.compile(r'^\s*(compute_analytical_stress)\s+(\S+)', re.IGNORECASE),
    'compute_heat_flux': re.compile(r'^\s*(compute_heat_flux)\s+(\S+)', re.IGNORECASE),
    'distributed_spline_storage': re.compile(r'^\s*(distributed_spline_storage)\s+(\S+)', re.IGNORECASE),
    'evaluate_work_function': re.compile(r'^\s*(evaluate_work_function)\s+(\S+)', re.IGNORECASE),
    'final_forces_cleaned': re.compile(r'^\s*(final_forces_cleaned)\s+(\S+)', re.IGNORECASE),
    'hessian_to_restart_geometry': re.compile(r'^\s*(hessian_to_restart_geometry)\s+(\S+)', re.IGNORECASE),
    'load_balancing': re.compile(r'^\s*(load_balancing)\s+(\S+)', re.IGNORECASE),
    'MD_clean_rotations': re.compile(r'^\s*(MD_clean_rotations)\s+(\S+)', re.IGNORECASE),
    'MD_restart': re.compile(r'^\s*(MD_restart)\s+(\S+)', re.IGNORECASE),
    'override_illconditioning': re.compile(r'^\s*(override_illconditioning)\s+(\S+)', re.IGNORECASE),
    'override_relativity': re.compile(r'^\s*(override_relativity)\s+(\S+)', re.IGNORECASE),
    'restart_relaxations': re.compile(r'^\s*(restart_relaxations)\s+(\S+)', re.IGNORECASE),
    'squeeze_memory': re.compile(r'^\s*(squeeze_memory)\s+(\S+)', re.IGNORECASE),
    'symmetry_reduced_k_grid': re.compile(r'^\s*(symmetry_reduced_k_grid)\s+(\S+)', re.IGNORECASE),
    'use_density_matrix': re.compile(r'^\s*(use_density_matrix)\s+(\S+)', re.IGNORECASE),
    'use_dipole_correction': re.compile(r'^\s*(use_dipole_correction)\s+(\S+)', re.IGNORECASE),
    'use_local_index': re.compile(r'^\s*(use_local_index)\s+(\S+)', re.IGNORECASE),
    'use_logsbt': re.compile(r'^\s*(use_logsbt)\s+(\S+)', re.IGNORECASE),
    'vdw_correction_hirshfeld': re.compile(r'^\s*(vdw_correction_hirshfeld)\s+(\S+)', re.IGNORECASE),
}

float_patterns = {
    'charge': re.compile(r'^\s*(charge)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
    'charge_mix_param': re.compile(r'^\s*(charge_mix_param)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
    'default_initial_moment': re.compile(r'^\s*(default_initial_moment)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
    'fixed_spin_moment': re.compile(r'^\s*(fixed_spin_moment)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
    'hartree_convergence_parameter': re.compile(r'^\s*(hartree_convergence_parameter)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
    'harmonic_length_scale': re.compile(r'^\s*(harmonic_length_scale)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
    'ini_linear_mix_param': re.compile(r'^\s*(ini_linear_mix_param)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
    'ini_spin_mix_parma': re.compile(r'^\s*(ini_spin_mix_parma)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
    'initial_moment': re.compile(r'^\s*(initial_moment)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
    'MD_MB_init': re.compile(r'^\s*(MD_MB_init)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
    'MD_time_step': re.compile(r'^\s*(MD_time_step)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
    'prec_mix_param': re.compile(r'^\s*(prec_mix_param)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
    'set_vacuum_level': re.compile(r'^\s*(set_vacuum_level)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
    'spin_mix_param': re.compile(r'^\s*(spin_mix_param)\s+([-+]?\d*\.?\d+)', re.IGNORECASE),
}

exp_patterns = {
    'basis_threshold': re.compile(r'^\s*(basis_threshold)\s+([-+]?\d*\.?\d+([eE][-+]?\d+)?)', re.IGNORECASE),
    'occupation_thr': re.compile(r'^\s*(occupation_thr)\s+([-+]?\d*\.?\d+([eE][-+]?\d+)?)', re.IGNORECASE),
    'sc_accuracy_eev': re.compile(r'^\s*(sc_accuracy_eev)\s+([-+]?\d*\.?\d+([eE][-+]?\d+)?)', re.IGNORECASE),
    'sc_accuracy_etot': re.compile(r'^\s*(sc_accuracy_etot)\s+([-+]?\d*\.?\d+([eE][-+]?\d+)?)', re.IGNORECASE),
    'sc_accuracy_forces': re.compile(r'^\s*(sc_accuracy_forces)\s+([-+]?\d*\.?\d+([eE][-+]?\d+)?)', re.IGNORECASE),
    'sc_accuracy_rho': re.compile(r'^\s*(sc_accuracy_rho)\s+([-+]?\d*\.?\d+([eE][-+]?\d+)?)', re.IGNORECASE),
    'sc_accuracy_stress': re.compile(r'^\s*(sc_accuracy_stress)\s+([-+]?\d*\.?\d+([eE][-+]?\d+)?)', re.IGNORECASE),
}

int_patterns = {
    'empty_states': re.compile(r'^\s*(empty_states)\s+(\d+)', re.IGNORECASE),
    'ini_linear_mixing': re.compile(r'^\s*(ini_linear_mixing)\s+(\d+)', re.IGNORECASE),
    'max_relaxation_steps': re.compile(r'^\s*(max_relaxation_steps)\s+(\d+)', re.IGNORECASE),
    'max_zeroin': re.compile(r'^\s*(max_zeroin)\s+(\d+)', re.IGNORECASE),
    'multiplicity': re.compile(r'^\s*(multiplicity)\s+(\d+)', re.IGNORECASE),
    'n_max_pulay': re.compile(r'^\s*(n_max_pulay)\s+(\d+)', re.IGNORECASE),
    'sc_iter_limit': re.compile(r'^\s*(sc_iter_limit)\s+(\d+)', re.IGNORECASE),
    'walltime': re.compile(r'^\s*(walltime)\s+(\d+)', re.IGNORECASE)
}
# TH: some of them seem unnecessary for our purposes and are complicated
#     to put into regex which is why i commented them out
list_patterns = {
    #'init_hess',
    'k_grid': re.compile(r'^\s*(k_grid)\s+(\d+)\s+(\d+)\s+(\d+)', re.IGNORECASE),
    'k_offset': re.compile(r'^\s*(k_offset)\s+(\d*\.?\d+)\s+(\d*\.?\d+)\s+(\d*\.?\d+)', re.IGNORECASE),
    #'MD_run',
    #'MD_schedule',
    #'MD_segment',
    #'mixer_threshold',
    'occupation_type': re.compile(r'^\s*(occupation_type)\s+(\S+)\s+(\d*\.?\d+)(?:\s+(\d+))?', re.IGNORECASE),
    #'output',
    #'cube',
    #'preconditioner',
    'relativistic':re.compile(r'^\s*(relativistic)\s+(\S+)\s+(\S+)(?:\s+(\d+))?', re.IGNORECASE),
    #'relax_geometry',
}

special_patterns = {
    'many_body_dispersion': re.compile(r'^\s*(many_body_dispersion)\s', re.IGNORECASE)
}
def f90_bool_to_py_bool(
    f90_bool:str
    )-> bool:
    
    if f90_bool.lower() == '.true.':
        return True
    elif f90_bool.lower() == '.false.':
        return False
    


aims_settings = {}
with open('control.in', 'r') as input_file:
    for line in input_file:
        
        if '#' in line:
            line = line.split('#')[0]
        
        for key, pattern in string_patterns.items():
            match = pattern.match(line)
            if match:
                aims_settings[match.group(1)] = match.group(2)
        for key, pattern in bool_patterns.items():
            match = pattern.match(line)
            if match:
                aims_settings[match.group(1)] = f90_bool_to_py_bool(match.group(2))
        for key, pattern in float_patterns.items():
            match = pattern.match(line)
            if match:
                aims_settings[match.group(1)] = float(match.group(2))
        for key, pattern in exp_patterns.items():
            match = pattern.match(line)
            if match:
                aims_settings[match.group(1)] = float(match.group(2))
        for key, pattern in int_patterns.items():
            match = pattern.match(line)
            if match:
                aims_settings[match.group(1)] = int(match.group(2))
        for key, pattern in list_patterns.items():
            match = pattern.match(line)
            if match:
                if key == 'k_grid':
                    aims_settings[match.group(1)] = [int(match.group(2)), int(match.group(3)), int(match.group(4))]
                if key == 'k_offset':
                    aims_settings[match.group(1)] = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
                if key == 'occupation_type':
                    if match.group(4) is not None:
                        aims_settings[match.group(1)] = [match.group(2), float(match.group(3)), int(match.group(4))]
                    else:
                        aims_settings[match.group(1)] = [match.group(2), float(match.group(3))]
                if key == 'relativistic':
                    if match.group(4) is not None:
                        aims_settings[match.group(1)] = [match.group(2), match.group(3), int(match.group(4))]
                    else:
                        aims_settings[match.group(1)] = [match.group(2), match.group(3)]
        for key, pattern in special_patterns.items():
            match = pattern.match(line)
            if match:
                if key == 'many_body_dispersion':
                    aims_settings[match.group(1)] = ''
                        
aims_settings['species_dir'] = "/project/home/p200243/tcp_software/aims_240507_library/FHIaims/species_defaults/defaults_2020/light"
ASI_path = "/project/home/p200243/tcp_software/aims_240507_library/libaims.240627.scalapack.mpi.so"
atoms1 = molecule('H2')
atoms2 = molecule('O2')

def init_via_ase(asi):
    from ase.calculators.aims import Aims
    calc = Aims(**aims_settings)
    calc.write_input(asi.atoms)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

color = 0 if rank == 0 else 1
sub_comm = comm.Split(color, rank)
comm.barrier()

if color == 0:
    atoms1.calc = ASI_ASE_calculator(
                ASI_path,
                init_via_ase,
                sub_comm,
                atoms1
        )

if color == 1:
    atoms2.calc = ASI_ASE_calculator(
            ASI_path,
            init_via_ase,
            sub_comm,
            atoms2
        )

comm.barrier()


if color == 0:
    atoms1.calc.calculate(atoms1)
    print(atoms1.calc.results)
    print(atoms1.get_positions())
    print(atoms1.get_atomic_numbers())
if color == 1:
    atoms2.calc.calculate(atoms2)
    print(atoms2.calc.results)

    print(atoms2.get_positions())
    print(atoms2.get_atomic_numbers())
#print(energy)
#atoms2.get_potential_energy()
