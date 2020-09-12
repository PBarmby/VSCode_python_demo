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
def test_add(a, b):
    return(a + b)
# %%
test_add(2,7)
# %%
def test_display(a, b, c = 3):
    print("a", a, "b", b, "c", c)
    return    

# %%
test_display(1,2)
# %%
test_display(1,2,c=7)
# %%
fahr_to_kelvin("forty-two")
# %%
def s(p):
    a = 0
    for v in p:
        a += v
    m = a / len(p)
    d = 0
    for v in p:
        d += (v - m) * (v - m)
    return numpy.sqrt(d / (len(p) - 1))
# %%
def std_dev(sample):
    sample_sum = 0
    for value in sample:
        sample_sum += value

    sample_mean = sample_sum / len(sample)

    sum_squared_devs = 0
    for value in sample:
        sum_squared_devs += (value - sample_mean) * (value - sample_mean)

    return numpy.sqrt(sum_squared_devs / (len(sample) - 1))
# %%
