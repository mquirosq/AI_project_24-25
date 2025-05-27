from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ai-project-24-25",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
    author="María Quirós Quiroga",
    author_email="marquiqui@alum.us.es",
    description="AI Project for 2024-2025",
    python_requires=">=3.7",
)