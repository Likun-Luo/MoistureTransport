# Executable release

Here, we note our experiences with generating the executable out of the python script `main.py`.
Executables have some pecularities and other oddities that one must be aware of when writing a python script/program!

## Some peculiarities to start with

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

## pyinstaller

Commands:

One-file version

```bash
pyinstaller --noconfirm --onefile --console --icon "./icons/favicon/b074e2c6121e95222f48af881040a0a2.ico/android-icon-36x36.png" --name "moisture_transport" --log-level "WARN" main.py
```

One-directory version

```bash
pyinstaller --noconfirm --onedir --console --icon "./icons/favicon/b074e2c6121e95222f48af881040a0a2.ico/android-icon-36x36.png" --name "moisture_transport_{VERSION}" --log-level "WARN" --add-data "./cfg:cfg/"  main.py
```
