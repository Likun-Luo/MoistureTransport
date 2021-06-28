import pathlib
from inspect import getsourcefile

#
a = getsourcefile(lambda:0)
path = pathlib.Path(a).absolute()
print(path)
