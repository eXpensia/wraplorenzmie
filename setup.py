from setuptools import setup
from setuptools import find_packages

setup(
    name='wraplorenzmie',
    author="Maxime Lavaud",
    author_email="lavaudmaxime.ml@gmail.com",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'ipympl==0.9.2',
        'jupyterlab==3.4.8',
        'matplotlib==3.6.1',
        'numpy==1.23.4',
        'scipy==1.9.2',
        'seaborn==0.11',
        'tqdm==4.64.1',
        'dill==0.3.5.1',
        'opencv-python==4.6.0.66',
        'h5py==3.7.0',
        'trackpy==0.5.0',
        'numba==0.56.3',
        'lmfit==1.0.3',
        'imageio==2.22.3',
        'streamlit==1.14.0',
        'streamlit-cropper==0.2.0',
        'imageio_ffmpeg==0.4.7',
    ],
    packages=find_packages(),
)
