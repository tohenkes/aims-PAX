# General

**Currently this version runs only on one core (except initial dataset procedure).**

- To use the code you just have to have the 
    - control.in, 
    - geometry.in,
    - mace_settings.yaml,
    - and active_learning_settings.yaml 
in the working directory and 'mpirun -n $CORES python run_whole_procedure.py'.

- Currently, the ability to read the aims settings in control.in is *very* limited (see handle_aims_settings() in procedures.py).
- The mace and AL settings should be self-explanatory with the comments and their names otherwise let me know if something is unclear.
- The script creates some folders and files these are:
    - checkpoints/: model checkpoints during training are saved here
    - data/: the initial and final datasets are saved here, as well as a dictionary that specifies the seeds for each ensemble member *(TODO: maybe move somewhere else)*
    - results/ *(TODO: rename)*: losses over time for each individual ensemble member
    - model/: saves the final model(s) here
    - initial_dataset.log: Log file for the creation of the initial dataset. Loss and errors are ensemble averages.
    - AL.log: Log file for the actual active learning procedure. Loss and errors are ensemble averages.





# ToDo

There is some bug in the active learning. The energy error is exploding andthe training becomes completely unstable. Either there is some problems with the units or data. For the latter it could be that the models generate crazy geometries during MD but I am not sure.

- [] energy error bug fix (see above)
- [] parallelism for FHI aims in AL procedure
- [] add more documentation


# Requirements

1. ASE
2. numpy
3. asi4py
4. mace
5. pyTorch
6. pyYaml

I think that's it. Please tell me if something missing.
