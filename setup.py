from setuptools import setup, Extension

setup(
    name="attention",
    version="0.01",
    description="fast attention merging module with ctypes",
    extras_require={"test": ["hypothesis"]},
    packages=['attention'],
    ext_modules=[Extension("", ["attentionmodule.c"])],
)
