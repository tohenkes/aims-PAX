from setuptools import setup, find_packages

setup(
    name='aims_PAX',
    version='0.1',
    author='Tobias Henkes',
    author_email='tobias.henkes@uni.lu',
    description='Active learning for FHI-aims',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tohenkes/aims_PAX',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.9',
    install_requires=[
        "numpy==1.26.4",
        "mace-torch==0.3.9",
        "asi4py==1.3.18",
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        "pyyaml==6.0.1",
        "ase==3.23.0"
    ],
    extras_require={
        'parsl': ['parsl==2024.12.16'],
    },
    entry_points={
        'console_scripts': [
            'aims_PAX=aims_PAX.cli.__main__:main',
            'aims_PAX-initial-ds=aims_PAX.cli.create_initial_ds:main',
            'aims_PAX-al=aims_PAX.cli.al_procedure_only:main',
            #"aims_PAX-scratch=aims_PAX.cli.scratch_train:main",
            #"aims_PAX-atomic-energies=aims_PAX.cli.get_atomic_energies:main",
            #"aims_PAX-test_ensemble=aims_PAX.cli.test_ensemble:main",
            #"aims_PAX-recalculate=aims_PAX.cli.recalculate_data:main",
        ],
    },
)