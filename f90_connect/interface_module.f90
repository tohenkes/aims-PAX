module interface_module
    use, intrinsic :: iso_c_binding
    implicit none

    interface
        subroutine call_python_function(energy, coords, forces, n_atoms, step_number, status) bind(C, name="call_python_function")
            use, intrinsic :: iso_c_binding
            integer(c_int), intent(in), value :: n_atoms
            integer(c_int), intent(in), value :: step_number
            real(c_double), intent(in), value :: energy
            real(c_double), intent(in), dimension(n_atoms,3) :: coords
            real(c_double), intent(in), dimension(n_atoms,3) :: forces
            integer(c_int), intent(out) :: status
        end subroutine call_python_function

        subroutine initialize_python() bind(C, name="initialize_python")
            use, intrinsic :: iso_c_binding
        end subroutine initialize_python

        subroutine finalize_python() bind(C, name="finalize_python")
            use, intrinsic :: iso_c_binding
        end subroutine finalize_python
    end interface
end module interface_module
