from setuptools import setup, find_packages

setup(
    name='image_classifier',
    version='1.0.0',
    author='siddhi',
    author_email='siddhikiran.bajracharya@gmail.com',
    description='A extendible generic image classifier',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'opencv-python',
        'pandas',
        'numpy',
        'scikit-learn'

    ],
    )
