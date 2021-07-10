import pathlib
import inspect
import sys

RESULTS_DIR = pathlib.Path("./results").absolute()
RESULTS_DIR.mkdir(exist_ok=True)

def get_module_path(local_function):
   ''' returns the module path without the use of __file__.  Requires a function defined
   locally in the module.
   from http://stackoverflow.com/questions/729583/getting-file-path-of-imported-module'''
   return pathlib.Path(inspect.getsourcefile(local_function)).absolute()
   #return os.path.abspath(inspect.getsourcefile(local_function))

module_path = get_module_path (lambda:0)
# checks whether one is running the script ('.py') or executable/frozen ('.exe') version
# from: https://stackoverflow.com/questions/404744/determining-application-path-in-a-python-exe-generated-by-pyinstaller/404750#404750
if getattr(sys, 'frozen', False):
    WORK_DIR = pathlib.Path(sys.executable).parent
else:
    WORK_DIR = module_path
    while WORK_DIR.name != "src":
        WORK_DIR = WORK_DIR.parent
    WORK_DIR = WORK_DIR.parent