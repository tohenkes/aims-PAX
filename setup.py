from setuptools import setup, find_packages

setup(
    name='FHI_AL',
    version='0.1',
    author='Tobias Henkes',
    author_email='tobias.henkes@uni.lu',
    description='Active learning for FHI-aims',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tohenkes/FHI_AL',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.9',
    install_requires=[
        'ase',
        'numpy',
        'mace-torch',
        'asi4py',
        'torch',
        'pyYaml',
    ],
    extras_require={
        'parsl': ['parsl'],
    },
    entry_points={
        'console_scripts': [
            'FHI_AL=FHI_AL.cli.__main__:main',
            'FHI_AL-initial-ds=FHI_AL.cli.create_initial_ds:main',
            'FHI_AL-al=FHI_AL.cli.al_procedure_only:main',
            "FHI_AL-scratch=FHI_AL.cli.scratch_train:main",
            "FHI_AL-atomic-energies=FHI_AL.cli.get_atomic_energies:main",
            "FHI_AL-test_ensemble=FHI_AL.cli.test_ensemble:main",
            "FHI_AL-recalculate=FHI_AL.cli.recalculate_data:main",
        ],
    },
)