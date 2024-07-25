# General

* The following is not supposed to be a tutorial. I am working on that. This is just so that we can work and review the code together.*

To use the code:
1. make sure the following files are in the working directory:
    - control.in, 
    - geometry.in,
    - mace_settings.yaml,
    - and active_learning_settings.yaml 
2. '''mpirun -n $CORES python run_whole_procedure.py'''.

- The mace and AL settings should be self-explanatory with the comments and their names otherwise let me know if something is unclear.
- The script creates some folders and files these are:
    - checkpoints/: model checkpoints during training are saved here
    - data/: the initial and final datasets are saved here, as well as a dictionary that specifies the seeds for each ensemble member *(TODO: maybe move somewhere else)*
    - results/ *(TODO: rename)*: losses over time for each individual ensemble member
    - model/: saves the final model(s) here
    - initial_dataset.log: Log file for the creation of the initial dataset. Loss and errors are ensemble averages.
    - AL.log: Log file for the actual active learning procedure. Loss and errors are ensemble averages.


# ToDo

- [x] energy error bug fix
- [x] parallelism for FHI aims in AL procedure
- [ ] analysis on parameters with naphtalene
- [ ] same with MD17/22?
- [ ] initial dataset train and MD in parallel
- [ ] AL train and MD in parallel
- [ ] multiple species at once
- [ ] add more documentation


# Requirements

1. ASE
2. numpy
3. asi4py
4. mace
5. pyTorch
6. pyYaml

I think that's it. Please tell me if something missing.
