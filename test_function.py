# %%
x = 2
print(x*2)
# %%
def testfunc(x):
    return(2*x)

# %%
y = testfunc(3)
print(y)
# %%
testfunc(3)
# %%
def fahr_to_celsius(temp_f):
    return((temp_f - 32) * (5/9))
# %%
fahr_to_celsius(98)
# %%
freeze = fahr_to_celsius(32)
# %%
print(freeze)
# %%
print('Boiling point:', fahr_to_celsius(212), "C")
# %%
