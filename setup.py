from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='GLMModels',
    version='0.0.1',
    description='Performs Latent Factor Analysis on Generalized Linear Mixed Models',
    keywords='variational inference, latent factor analysis, count data, bayesian inference, generalized linear mixed models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Ananyapam7/GLMModels',
    author='Ananyapam De',
    author_email='ananyapam7@gmail.com',
    py_modules=['LFA_SVI', 'sim_counts'],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'tqdm',
        'torch'
    ],
    package_dir={'': 'src'},
    classifiers=[
        # see https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Machine Learning',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    extras_require={
        'dev': [
            'pytest>=3.7',
            'pandas',
            'numpy',
            'matplotlib',
            'scikit-learn',
            'check-manifest',
            'twine',
            'sphinx',
            'coverage',
        ],
    },
)