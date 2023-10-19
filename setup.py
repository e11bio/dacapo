from setuptools import setup, find_packages

setup(
    name="dacapo-ml",
    description="Framework for easy composition of volumetric machine learning jobs.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    version="0.1",
    url="https://github.com/funkelab/dacapo",
    author="Jan Funke, Will Patton",
    author_email="funkej@janelia.hhmi.org, pattonw@janelia.hhmi.org",
    license="MIT",
    packages=find_packages(),
    entry_points={"console_scripts": ["dacapo=dacapo.cli:cli"]},
    include_package_data=True,
    package_data={"dacapo": ["py.typed"]},
    install_requires=[
        "numpy",
        "pyyaml",
        "zarr",
        "cattrs",
        "pymongo",
        "tqdm",
        "simpleitk",
        "lazy-property",
        "neuroglancer",
        "torch",
        "fibsem_tools",
        "attrs",
        "bokeh",
        "numpy-indexed>=0.3.7",
        "daisy>=1.0",
        "funlib.math>=0.1",
        "funlib.geometry>=0.2",
        "mwatershed>=0.1",
        "funlib.persistence>=0.1",
        "funlib.evaluate @ git+https://github.com/pattonw/funlib.evaluate",
        # "gunpowder>=1.3",
        "gunpowder @ git+https://github.com/funkelab/gunpowder@skip_node",
        "lsds>=0.1.3",
    ],
)
