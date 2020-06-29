from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="tranql-jupyter",
    description="TranQL bindings for Jupyter/IPython",
    include_package_data=True,
    packages=find_packages(),
    install_requires=requirements
)
