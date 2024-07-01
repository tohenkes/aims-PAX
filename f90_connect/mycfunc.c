#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

int initialize_python() {
    // Initialize the Python interpreter
    Py_Initialize();

    // Initialize NumPy
    import_array1(0);
    return 0;
}


void finalize_python() {
    // Finalize the Python interpreter
    Py_Finalize();
}

int call_python_function(double energy, double *coords, double *forces, int n_atoms, int step_number, int *status) {
    // Add the current directory to the Python module search path
    
    long return_val; 
    return_val = -1;
    // PyRun_SimpleString("import sys; sys.path.append('/home/tobias/Uni/Promotion/Research/aims_MLFF/active_learning/FHI_AL/f90_connect')");
    PyRun_SimpleString("import sys; sys.path.append('/home/thenkes/Documents/Uni/Promotion/Research/aims_MLFF/active_learning/FHI_AL/f90_connect')");
    // Load the Python module
    PyObject *pName = PyUnicode_DecodeFSDefault("mymodule");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        // Get the reference to the Python function
        PyObject *pFunc = PyObject_GetAttrString(pModule, "my_function");

        // Check if the function is callable
        if (pFunc && PyCallable_Check(pFunc)) {
            // Create a NumPy array from the C array
            npy_intp dims[2] = {n_atoms, 3};
            //PyObject *pArray = (3, dims, NPY_DOUBLE, (void *)arr);

            PyObject *pCoords = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void*)coords);
            PyObject *pForces = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void*)forces);

            if ((pCoords == NULL) || (pForces == NULL)) {
                PyErr_Print();
                fprintf(stderr, "Failed to create NumPy array\n");
                return -1;
            }
            PyObject *pArgs = PyTuple_Pack(4, PyFloat_FromDouble(energy),pCoords, pForces, PyLong_FromLong(step_number));
            PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
            *status = PyLong_AsLong(pValue);
            // Check for errors
            if (pValue != NULL) {
                Py_DECREF(pValue);
            } else {
                PyErr_Print();
                fprintf(stderr, "Call to Python function failed\n");
            }

            Py_DECREF(pArgs);
            Py_DECREF(pCoords);
            Py_DECREF(pForces);
            Py_XDECREF(pFunc);
        } else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"my_function\"\n");
        }

        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"mymodule\"\n");
    }
    return return_val;
}
