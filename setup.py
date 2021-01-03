from setuptools import setup

setup(
    name="attention",
    version="0.0.1",
    description="fast attention merging module with ctypes",
    install_requires=['numpy'],
    extras_require={"test": ["hypothesis"]},
    packages=["attention"],
)
