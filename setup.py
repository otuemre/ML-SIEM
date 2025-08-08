from pathlib import Path
from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    Read requirements.txt and return a clean list of packages.
    - Ignores comments and blank lines
    - Removes '-e .' if present
    """

    req_path = Path(file_path)
    requirements: List[str] = []
    if req_path.exists():
        for line in req_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            requirements.append(line)

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="ml-siem",
    version="0.0.1",
    description="End-to-end ML SIEM: ingestion → FE → transform → autoencoder anomaly detection",
    author="Emre OTU",
    author_email="dev.emre17@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.9",
)
