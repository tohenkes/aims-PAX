"""
The module tests utils from an atomate2 subsystem of aims-PAX.
"""
from jobflow import job, run_locally
import numpy as np
from aims_PAX.atomate2.random import RandomState


def test_random_state():
    rng = RandomState(1234)
    # create 5 numbers to initialize the RNG
    rng.random_integers(0, 100, 5)
    rng_dict = rng.as_dict()
    rng_restored = RandomState.from_dict(rng_dict)
    assert np.allclose(
        rng.random_integers(0, 100, 5),
        rng_restored.random_integers(0, 100, 5)
    )


def test_random_state_job(clean_dir):
    @job
    def sample_job(rng: RandomState, n_samples: int):
        samples = rng.random_integers(0, 100, n_samples)
        return {"samples": samples.tolist()}

    # Call it directly
    rng = RandomState(1234)
    rng.random_integers(0, 100, 5)  # advance state first

    my_job = sample_job(rng, n_samples=10)
    output = run_locally(my_job, create_folders=True)
    uuid = list(output.keys())[0]
    assert output[uuid][1].output["samples"] == [24, 15, 49, 23, 26, 30, 43, 30, 26, 58]