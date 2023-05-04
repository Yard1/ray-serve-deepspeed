from setuptools import find_packages, setup


def readme():
    with open("README.md", encoding="utf8") as f:
        README = f.read()
    return README


setup(
    name="rayviary",
    version="0.0.1",
    description="Rayviary",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Antoni Baum",
    author_email="antoni@anyscale.com",
    packages=find_packages(include=["rayviary*"]),
    include_package_data=True,
    python_requires=">=3.7",
)
