# Divi

**Divi** is a Python library for generating quantum programs at scale. It allows users to parallelize quantum workloads, manage hardware or simulator execution, and interact with cloud-based compute clusters—all through a simple and flexible API.

Divi is designed to allow researchers, students, and enterprises to deploy quantum workloads efficiently across hybrid and distributed environments.

---

> [!IMPORTANT]
> Divi is under active development and in early stages. Users should expect frequent changes that are likely to be incompatible with previously published versions.

## 🚀 Features

- **Smart Job Parallelization**: Automatically parallelizes your quantum programs based on task structure.
- **Cloud Execution**: Send jobs to local or remote clusters with minimal configuration.
- **Extensible API**: Easily integrate with your existing quantum stack.
- **Lightweight & Modular**: Install and use only what you need.

---

## 📦 Installation

Divi can be easily installed from Pypi

```bash
pip install qoro-divi
```

## 📚 Documentation

- Full documentation is available at: <https://qoroquantum.github.io/divi/>
- Tutorials can be found in the `tutorials` folder.

## 🧪 Testing

To run the test suite:

```bash
# Run all tests
pytest

# Run only API tests (requires API token)
pytest --run-api-tests
```

**Note:** Some tests require a Qoro API token to test the cloud REST API. Set the `QORO_API_KEY` environment variable or use the `--api-token` option. For local development, you can create a `.env`.
