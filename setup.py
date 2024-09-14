from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="real_esrgan_generator",
    version="0.1.0",
    author="黃毓峰",
    author_email="a288235403@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IDK-Silver/dsrnet-detector",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        'basicsr>=1.4.2',
        'facexlib>=0.2.5',
        'gfpgan>=1.3.5',
        'numpy',
        'opencv-python',
        'Pillow',
        'torch>=1.7',
        'torchvision',
        'tqdm',
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
)