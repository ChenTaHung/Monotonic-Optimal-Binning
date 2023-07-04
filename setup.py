from setuptools import setup, find_packages

setup(
    name = 'MOBPY',
    version = '1.0',
    packages = find_packages(),
    install_requires = [
        'pandas',
        'numpy',
        'matplotlib >= 3.7.0',
        'os',
        'scipy >= 1.7.0'
    ],
    author=['Chen, Ta-Hung', 'Tsai, Yu-Cheng'],
    author_email= 'denny20700@gmail.com',
    description='Monotonic Optimal Binning is a statistical approach to transform continuous variables into optimal and monotonic categorical variables.',
    url = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning'
    
)