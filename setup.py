from setuptools import setup, find_packages

setup(
        name="vanilla_attention",
        version="0.1",
        packages=find_packages(),
        install_requires=["torch", "numpy"],
        author="Mateo del Rio",
        description="Vanilla implementation of Attention Mechanism (Vaswani et al., 2017)"
        )
