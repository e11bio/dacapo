from setuptools import setup, find_packages

setup(
    name="dacapo",
    version="0.1",
    url="https://github.com/funkelab/dacapo",
    author="Jan Funke, Will Patton",
    author_email="funkej@janelia.hhmi.org, pattonw@janelia.hhmi.org",
    license="MIT",
    packages=find_packages(),
    entry_points={"console_scripts": ["dacapo=scripts.dacapo:cli"]},
    include_package_data=True,
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
        "daisy @ git+https://github.com/funkelab/daisy",
        "funlib.math @ git+https://github.com/funkelab/funlib.math@0c623f71c083d33184cac40ef7b1b995216be8ef",
        "funlib.evaluate @ git+https://github.com/pattonw/funlib.evaluate",
        "funlib.geometry @ git+https://github.com/funkelab/funlib.geometry@cf30e4d74eb860e46de40533c4f8278dc25147b1",
        "cremi @ git+https://github.com/cremi/cremi_python@python3",
        "gunpowder @ git+https://github.com/funkey/gunpowder@v1.3-dev",
        "lsd @ git+https://github.com/pattonw/lsd@no-convenience-imports",
    ],
)
