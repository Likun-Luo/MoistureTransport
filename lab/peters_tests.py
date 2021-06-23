# %%
from dataclasses import dataclass, asdict, field
import numpy as np
import matplotlib.pyplot as plt
# %%
@dataclass
class Candle:
    L:int = 10
    flame:int = 1
    # iter_dict:dict = field(default_factory=dict)

    # def __post_init__(self):
    #     self.iter_dict = asdict(self)

    def getAttrCount(self):
        di = self.__dir__()
        di = [var for var in di if not callable(var)]
        count = sum([1 for att in di if "__" not in att])
        print(di)
        return count

    def __iter__(self):
        dicte = asdict(self)
        print(dicte)
        return iter(dicte)

    def __getitem__(self, item):
        return self.__getattribute__(item)

    # def __next__(self):
    #     yield self.iter_dict

# %%
a = Candle()
a
# %%
for i in a:
    print(i, ": ", a[i])
# %%
next(a)
# %%
tmax = 1000
t = np.arange(0,tmax,1)

w = 10 * np.sqrt(t)
w_last = w[-1] * np.ones_like(w)
w = np.append(w, w_last)
t = np.arange(0,2*tmax,1)
# %%
fig, ax = plt.subplots(2, figsize=(8,6))
ax[0].plot(t, w)
ax[0].set_xlabel("t")
ax[1].plot(np.sqrt(t), w)
ax[1].set_xlabel("sqrt(t)")

# %%
volume = np.arange(0, 1000, 1)
volume

# %%timeit -n 1000 -r 10
%%timeit -n 1000 -r 10
with np.nditer([volume[:-2], volume[1:-1], volume[2:]], op_flags=['readwrite'], flags=["f_index"], order="C") as it:
    for a,b,c in it:
        idx = it.index + 1
        b = b + (c - a) / 2
        # print("idx: ", idx, " --> (", a, ", ", b, ", ", c, ")")
# %%
%%timeit -n 1000 -r 10
for idx in range(1, len(volume) -1):
    # print(idx)
    volume[idx] += (volume[idx+1] - volume[idx-1]) / 2
# %%
def format_time_scientific(num):
    if time_in_s>86399.5:
        return f'{time_in_s/86400:2.1f}'+ "d"
    if time_in_s>3599.5:
        return f'{time_in_s/3600:2.1f}'+ "h"
    if time_in_s>59.5:
        return f'{time_in_s/60:2.1f}'+ "m"
    multiplier = 1000
    for unit in ("s", "ms","µs", "fs"):
        if time_in_s > 0.995:
            if time_in_s > 9.95:
                if time_in_s > 99.5:
                    return f'{time_in_s:3.0f}'+ unit
                return f'{time_in_s:2.1f}'+ unit
            return f'{time_in_s:1.2f}'+ unit
        time_in_s *= multiplier
    return f'{time_in_s:3.3f}'


# %%
print(format_time_scientific(3601))
print(format_time_scientific(3599))
print(format_time_scientific(61))
print(format_time_scientific(59))
print(format_time_scientific(3))
print(format_time_scientific(0.901))
print(format_time_scientific(0.09))
print(format_time_scientific(0.009))
print(format_time_scientific(0.0009))
print(format_time_scientific(0.00009))
print(format_time_scientific(0.000009))
# %%
24*3600
# %%
from dataclasses import dataclass
# %%
@dataclass
class Test:
    a:int = 1

    @staticmethod
    def give(b):
        return b
# %%
test = Test(3)
# %%
test.give(test.a)
# %%
Test.give(3)
# %%
