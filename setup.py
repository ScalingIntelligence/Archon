from setuptools import setup, find_packages
from pathlib import Path
current_directory = Path(__file__).parent
VERSION = '0.0.4' 
DESCRIPTION = "Create and Benchmark LLM Chains with JSON"
LONG_DESCRIPTION = (current_directory / "README.md").read_text()

setup(
    name="archon-ai",
    version=VERSION,
    author="Shlok Natarajan",
    author_email="shlok.natarajan@gmail.com",
    url="https://github.com/jonsaadfalcon/Archon",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "anthropic>=0.31.2",
        "datasets>=2.20.0",
        "groq>=0.9.0",
        "huggingface-hub>=0.24.2",
        "loguru>=0.7.2",
        "numpy",
        "openai>=1.37.1",
        "pandas>=2.2.2",
        "plotly>=5.23.0",
        "requests>=2.32.3",
        "shortuuid>=1.0.13",
        "tabulate>=0.9.0",
        "tiktoken>=0.7.0",
        "together>=1.2.3",
        "tokenizers>=0.19.1",
        "tqdm>=4.66.4",
        "transformers",
        "torch"
    ]
)