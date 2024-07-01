program callcfromfortran
    use interface_module
    implicit none

    integer(c_int) ::  n_atoms, step_number,i, status
    real(c_double) :: energy
    real(c_double), dimension(:,:), allocatable :: coords, forces
    integer(c_int) :: result_val
    logical(c_bool) :: stop_md
    n_atoms = 3
    energy = -25.203512
    status = 0
    ! Allocate array
    allocate(coords(n_atoms, 3))
    allocate(forces(n_atoms, 3))
    coords = reshape([(i, i=1, n_atoms*3)], shape(coords))
    forces = reshape([(i, i=1, n_atoms*3)], shape(forces))
    ! Initialize Python interpreter
    call initialize_python()
    ! Call the C function multiple times
    do step_number = 0, 1000
        call call_python_function(energy, coords, forces, n_atoms, step_number, status)
        if (status == 1) then
            stop
        end if
    end do

    ! Finalize Python interpreter
    call finalize_python()

end program callcfromfortran
