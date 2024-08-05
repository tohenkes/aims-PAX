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
    entry_points={
        'console_scripts': [
            'FHI_AL=FHI_AL.__main__:main',
        ],
    },
)