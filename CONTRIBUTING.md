# Contributing to Divi

Thank you for considering contributing to Divi!

Divi is an open-source Python library designed for distributed quantum program execution. Whether you’re here to fix bugs, improve documentation, build new features, or test our code on different platforms, your contributions are highly appreciated.

--- 

## What You Can Contribute

Here are some common ways to get involved:
- Examples written in the Divi framework 
- Bug fixes
- New features or enhancements
- Benchmarking and testing improvements
- Backend integrations
- Local simulator integrations

## Getting Started

### 1. Fork the Repository

Click “Fork” at the top right of [the main repository](https://github.com/qoroquantum/divi) and clone your fork:

```bash
git clone https://github.com/your-username/divi.git
cd divi
```

### 2. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```
Or if using `poetry`:
```bash
poetry install
poetry shell
```

### 3. Workflow
1. Create a new branch with a descriptive name, for example: `git checkout -b feature/implementation-of-qaoa`
2. Make the changes.
3. Create at least one unit test (if a feature).
4. Ensure other tests are passing.
5. Format the code with `black`: `poetry run black .`
6. Push your code and create a pull request.

## Code Style
We use [Black](https://github.com/psf/black) for formatting. 



