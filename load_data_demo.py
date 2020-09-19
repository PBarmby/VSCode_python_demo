# %%
import numpy as np
# %%
data = np.loadtxt('inflammation-01.csv', delimiter=',')
# %%
data
# %%
print(data)
# %%
print(type(data))
# %%
print(data.shape)
# %%
print(data[0,0])
# %%
print(data[-1,-1])
# %%
print(data[0,:])
# %%
print(data[:,0])
# %%
import matplotlib.pyplot as plt
# %%
image = plt.imshow(data)
# %%
plt.plot(data[0,:])
# %%
plt.scatter(data[0,:],data[1,:])
# %%
import pandas as pd
# %%
import seaborn as sns
# %%
df = pd.read_csv('gapminder_gdp_europe.csv')
# %%
df.info()
# %%
sns.relplot(x='gdpPercap_1952',y='gdpPercap_2007',data=df)
# %%
plt.scatter(x=df['gdpPercap_1952'],y=df['gdpPercap_2007'])
# %%
