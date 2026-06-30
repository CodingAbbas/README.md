#!/usr/bin/env python
# coding: utf-8

# ## 1) Import Libraries

# In[239]:


import pandas as pd
import numpy as np
import wbdata
import datetime
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# ## 2) Data Collection

# ### Define World Bank indicators

# In[240]:


indicators = {

    "SP.DYN.TFRT.IN": "fertility_rate",
    "NY.GDP.PCAP.KD.ZG": "GDP_per_capita"

}


# In[241]:


countries = [

    "DEU", "FRA", "ITA", "ESP", "POL", "SWE", "NLD", "BEL", "GRC", "PRT",
    "CZE", "HUN", "ROU", "AUT", "CHE", "NOR", "DNK", "FIN", "SVK", "BGR",
    "HRV", "SVN", "LTU", "LVA", "EST", "IRL", "GBR", "UKR", "SRB", "ALB",

]


# ### Pull data from the World Bank API

# In[242]:


df = wbdata.get_dataframe (indicators, country = countries)

print(df.head(10))


# ## 3) Data Cleaning

# ### Check summary statistics to identify any data quality issues

# In[243]:


df.describe()


# ### Check missing value counts per variable

# In[244]:


df.isnull().sum()


# ### Drop all rows with missing values

# In[245]:


df = df.dropna()


# ### Reset index to bring country and date out as columns

# In[246]:


df_reset = df.reset_index()


# ### Convert date column to datetime format

# In[247]:


df_reset["date"] = pd.to_datetime (df_reset["date"])


# ### Trim to the 1990 to 2020 study period

# In[248]:


df_model = df_reset[

    (df_reset["date"].dt.year >= 1990) &
    (df_reset["date"].dt.year <= 2020)

].copy()


# ### Add a numeric year column for use in the fixed effects model

# In[249]:


df_model["year"] = df_model["date"].dt.year


# In[250]:


print(f"Date range: {df_model['year'].min()} to {df_model['year'].max()}")


# In[251]:


print(f"Observations: {df_model.shape[0]}")


# In[252]:


print(f"Countries: {df_model['country'].nunique()}")


# ## 4) Exploratory Data Analysis

# ### Chart 1: Fertility rate trends across all 30 countries with 2.1 replacement threshold

# In[253]:


plt.figure (figsize = (14, 6))

for country in df_model["country"].unique():
    country_data = df_model[df_model["country"] == country]
    plt.plot(country_data["date"], country_data["fertility_rate"], alpha = 0.5)

plt.axhline (y = 2.1, color = "red", linestyle = "--", linewidth = 1.5, label = "Replacement Level (2.1)")

plt.title ("Fertility Rate Across European Countries (1990-2020)")
plt.xlabel ("Year")
plt.ylabel ("Fertility Rate")

plt.legend()
plt.tight_layout()
plt.show()


# ### Chart 2: Rank 30 countries by average fertility rate 

# In[254]:


avg_fertility = df_model.groupby("country")["fertility_rate"].mean().sort_values()

plt.figure (figsize = (12, 8))
plt.barh (avg_fertility.index, avg_fertility.values, color = "steelblue")
plt.axvline (x = 2.1, color = "red", linestyle = "--", linewidth = 1.5, label = "Replacement Level (2.1)")

plt.title ("Average Fertility Rate by Country (1990-2020)")
plt.xlabel ("Average Fertility Rate")
plt.ylabel ("Country")

plt.legend()
plt.tight_layout()
plt.show()


# ### Chart 3: Raw relationship between fertility rate and GDP per capita

# In[255]:


plt.figure (figsize=(10, 6))
plt.scatter (df_model["fertility_rate"], df_model["GDP_per_capita"], alpha = 0.3, color = "steelblue")
plt.axvline (x = 2.1, color = "red", linestyle = "--", linewidth = 1.5, label = "Replacement Level (2.1)")

plt.title ("Fertility Rate vs GDP Per Capita (1990-2020)")
plt.xlabel ("Fertility Rate")
plt.ylabel ("GDP Per Capita Growth (%)")

plt.legend()
plt.tight_layout()
plt.show()


# ### Chart 4: Correlation heatmap between fertility rate and GDP growth

# In[256]:


corr_matrix = df_model[["fertility_rate", "GDP_per_capita"]].corr()

fig, ax = plt.subplots(figsize = (6, 5))
cax = ax.matshow(corr_matrix, cmap = "coolwarm")
fig.colorbar(cax)

ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="left")
ax.set_yticklabels(corr_matrix.columns)

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")

plt.title ("Correlation Heatmap (1990-2020)", pad=20)
plt.tight_layout()
plt.show()


# ## 5) Panel Data Modelling (Fixed effects OLS regression)

# In[257]:


model_gdp = smf.ols(
    "GDP_per_capita ~ fertility_rate + C(country) + C(year)",
    data = df_model
).fit()

print(model_gdp.summary())


# ## 6) Results Summary

# In[258]:


coef = round (model_gdp.params["fertility_rate"], 4)
pval = round (model_gdp.pvalues["fertility_rate"], 3)
rsq = round (model_gdp.rsquared, 3)
obs = int (model_gdp.nobs)

print("Fixed Effects Panel Regression Results")
print("Dependent Variable: GDP Per Capita Growth")

print("=" * 45)
print(f"Fertility Rate Coefficient : {coef}")
print(f"P-Value                    : {pval}")
print(f"R-Squared                  : {rsq}")
print(f"Observations               : {obs}")
print(f"Countries                  : 30")
print(f"Period                     : 1990 to 2020")
print("=" * 45)
print(f"Significant at 1% level    : {'Yes' if pval < 0.01 else 'No'}")

