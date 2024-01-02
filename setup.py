from setuptools import setup, find_packages

setup(
    name="flossy",
    version='1.0.0',
    author="Stefano Garcia",
    author_email="stefano.rgc@gmail.com",
    description="A package for interactively analyzing period spacing patterns.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "numba",
        "astropy",
        "PyQt5",
    ],
    python_requires='>=3.10',
)
