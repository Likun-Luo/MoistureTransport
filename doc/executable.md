# Executable release

Here, we note our experiences with generating the executable out of the python script `main.py`.
Executables have some pecularities and other oddities that one must be aware of when writing a python script/program!

## Why compile / build an executable?

There may be different reasons to do so, but mostly it will be either to protect your source code (kind of pointless for a FOSS project), to make your program more portable (don't need to take care of the environment, etc.) or to make it more accessible to non-programming literate people (which is the case here!).

If you expect better performance, than you might get lucky but don't expect any huge leaps - you're better of looking elsewhere.
Consider techniques like JIT compilation (Numba, JAX, ...), refactoring your code or utilizing other, better libraries.

## pyInstaller

Ultimately, we chose to go with the standard `pyinstaller`.
There is also a simple GUI version available called `auto-py-to-exe`(install via pip), which you can use to test the different available options of `pyinstaller`.

Some peculiarities to start with:

### exit()

One must use `from sys import exit` if one wants to use the exit-function - in script version, this is not necessary - otherwise the executable will terminate with the following error: `NameError: name 'exit' is not defined`.

### Getting the path to the executables directory

This is a bit more difficult since all the normal ways to get a script-files names and path do not work in executable format.

However, the following does work (adapted from [stackoverflow](https://stackoverflow.com/questions/404744/determining-application-path-in-a-python-exe-generated-by-pyinstaller/404750#404750)):

```python
if getattr(sys, 'frozen', False):
    WORK_DIR = pathlib.Path(sys.executable).parent
else:
    ...
```

## Building with pyinstaller

Commands:

One-file version

```bash
pyinstaller --noconfirm --onefile --console --icon "./icons/favicon/b074e2c6121e95222f48af881040a0a2.ico/android-icon-36x36.png" --name "moisture_transport" --log-level "WARN" main.py
```

One-directory version

```bash
pyinstaller --noconfirm --onedir --console --icon "./icons/favicon/b074e2c6121e95222f48af881040a0a2.ico/android-icon-36x36.png" --name "moisture_transport_{VERSION}" --log-level "WARN" --add-data "./cfg:cfg/"  main.py
```

---
---

## Different compilers we considered, but did not use in the end

We did consider some other compilers.
They have different pros and cons (such as e.g. cross plattform compilation: compile on one OS for other OSs).

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

Build time: 22min [Macbook Pro 2019 (Intel i7 2GHz)]

### Plugins

If you want to use/compile with certain packages or enable certain functionalities, you need to tell Nuitka to enable them, e.g.

numpy:

- `--plugin-enable=numpy`

PyLint/PyDev annotations for warnings:

- `--plugin-enable=pylint-warnings`

### Python command line flags

For passing things like -O or -S to Python, to your compiled program, there is a command line option name --python-flag= which makes Nuitka emulate these options.

The most important ones are supported, more can certainly be added.
