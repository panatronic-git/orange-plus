import os
from setuptools import setup

# rootdir = os.path.dirname(__file__)

INSTALL_REQUIRES = sorted(set(
    line.partition('#')[0].strip()
    for line in open('requirements.txt', 'r')
) - {''})

# User-friendly description from description.rst
with open("Description.rst", "r") as ufd:
    long_description = ufd.read()

setup(
    name = "Orange Plus",
    version = "0.1.0",
    description = "Orange3 custom widgets add-on.",
    url = "https://github.com/panatronic-git/orange-plus",
    author = "Panagiotis Papadopoulos",
    author_email = "'Panagiotis Papadopoulos' <panatronic@outlook.com>",
    license = "CC",
    download_url = "https://github.com/panatronic-git/orange-plus.git",
    keywords = "SMOTE,OPTICS,KDE-2D,data mining,orange3 add-on",
    platforms = "any",
    install_requires=INSTALL_REQUIRES,
    packages = ["orangeplus"],
    package_data = {"orangeplus": ["icons/*.svg"]},
    classifiers = [
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Data Mining",
        "Programming Language :: Python :: 3.6",
        "Environment :: Conda :: Orange3",
        "Environment :: Plugins",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    # Declare orangeplus package to contain widgets for the "Orange Plus" category
    entry_points = {"orange.widgets": "Orange Plus = orangeplus"},
    long_description = long_description,
    long_description_context_type = "text/markdown",
)