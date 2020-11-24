from setuptools import setup

setup(
    name = 'musico',
    python_requires = '==3.7.9',
    version = '0.10',
    packages = [
        'musico.emotions.utils', 'musico.emotions.models', 'musico.emotions.demo',
        'musico.instructions',
    ],
    long_description=open('README.md').read(),
    install_requires=[
        'h5py==2.10.0',
        'numpy==1.18.0',
        'scipy',
        'pandas',
        'matplotlib',
        'imageio',
        'opencv-python==4.4.0.46',
        'Keras==2.4.3',
        'tensorflow==2.3.1',
        'xlrd==1.2.0',
        'openpyxl==3.0.5',
    ],
    include_package_data = True,
)
