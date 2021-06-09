# Compilation

Latest build: 0.2

Build time: 22min [Macbook Pro 2019 (Intel i7 2GHz)]

## Compilers

Here is a list of compilers that we considered:

[nuitka](https://nuitka.net/)

- Nuitka is a Python compiler written in Python.
It is fully compatible with Python2 (2.6, 2.7) and Python3 (3.3 - 3.9).
You feed Nuitka your Python app, it does a lot of clever things, and spits out an executable or extension module.
Nuitka is distributed under the Apache license.

[Cython](https://cython.org/)

- Cython is an optimising static compiler for both the Python programming language and the extended Cython programming language (based on Pyrex). It makes writing C extensions for Python as easy as Python itself.

## Nuitka

Latest command used for compilation:

```bash
python -m nuitka --standalone --plugin-enable=numpy --plugin-enable=pylint-warnings --python-flag=-O --output-dir=build/ main.py
```

### Plugins

If you want to use/compile with certain packages or enable certain functionalities, you need to tell Nuitka to enable them, e.g.

numpy:

- `--plugin-enable=numpy`

PyLint/PyDev annotations for warnings:

- `--plugin-enable=pylint-warnings`

### Python command line flags

For passing things like -O or -S to Python, to your compiled program, there is a command line option name --python-flag= which makes Nuitka emulate these options.

The most important ones are supported, more can certainly be added.

