from setuptools import setup, find_packages

setup(
    name="crater-classifier",
    version="0.1",
    description="Binary CNN image classifier for cratered vs non-cratered terrain",
    author="Matt Colborn",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.12",
    install_requires=[
        "torch",
        "torchvision",
        "matplotlib",
        "pillow",
    ],
)
