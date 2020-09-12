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
def celsius_to_kelvin(temp_c):
    return(temp_c + 273.15)
# %%
celsius_to_kelvin(100)
# %%
def fahr_to_kelvin(temp_f):
    """Returns the conversion from Fahrenheit to Kelvin
          
       Usage: 
       fahr_to_kelvin(212) = 373.15

    """
    temp_c = fahr_to_celsius(temp_f)
    temp_k = celsius_to_kelvin(temp_c)
    return(temp_k)
# %%
fahr_to_kelvin(212)
# %%
fahr_to_kelvin(32)
# %%
help(fahr_to_kelvin)
# %%
help(print)
# %%
