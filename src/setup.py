from setuptools import find_packages, setup

with open("../README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='outlier_detection',
    version='0.0.1',
    author='Max Luebbering, Rajkumar Ramamurthy, Michael Gebauer',
    description="Outlier detection package",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=["torch",
                      "torchvision",
                      "flair",
                      "h5py",
                      "torchtext",
                      "tqdm",
                      "nltk >=3.4.5",
                      "flair",
                      "datastack",
                      "mlgym",
                      "outlier-hub"
                      ],
    python_requires=">=3.7"
)
