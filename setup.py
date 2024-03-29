from setuptools import setup, find_packages

try:
    # Attempt to import google.colab module
    import google.colab

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    # if shit is run colab we need to dribble some paths
    readme_path = "/content/drive/MyDrive/applied_DL/README.md"
else:
    readme_path = "README.md"

with open(readme_path, "r") as fh:
    long_description = fh.read()

setup(
    name="TimeSeriesDiffusion",
    version="1.0.0",
    description="A Python package for implementing diffusion models and anomaly detection on time series data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="sfz",
    author_email="fabian_zheng@yahoo.de",
    url="https://github.com/sf-zhg/APPLIED_DL",
    packages=find_packages(),
)
