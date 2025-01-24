from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    reqs = f.read().split("\n")

with open("requirements_dev.txt") as f:
    reqs_dev = f.read().split("\n")

setup(
    name="divi",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=reqs,
    extras_require={"graph": ["PyMetis==2023.1.1"], "dev": reqs_dev},
)
