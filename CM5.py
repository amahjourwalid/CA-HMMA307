#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Packages dont nous aurons besoin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact
from download import download
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Chargement des données
url = "http://josephsalmon.eu/enseignement/datasets/Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv"
path_target = "./Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv"
download(url, path_target, replace=False)

# Renomer le tableau de données et affichage de ses dimensions
df_poccitanie = pd.read_csv("Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv")
print(df_poccitanie.shape)

# Analyse du tableau:
df_poccitanie.head()  # affichage du début du tableau
df_poccitanie.columns  # affichage des variables
df_poccitanie['valeur_originale'].unique()  # affichage des valeurs originales
polluant = df_poccitanie['polluant'].unique()  # affichage des polluants
villes = df_poccitanie['nom_com'].unique()  # affichage des villes

# Ajout d'une colonne de temporalité "day" au tableau 
df_poccitanie['day'] = pd.to_datetime(df_poccitanie['date_debut'])
df_poccitanie.columns

# Histogramme des taux de chaque polluant
plt.figure(figsize=(8, 8))
plt.hist(df_poccitanie['polluant'], density=True, bins=50)
plt.xlabel("Polluant")
plt.ylabel("Taux du polluant")
plt.title("Taux des polluants")  # O3 est le plus élevé

# Création d'un nouveau tableau avec le polluant O3 uniquement
df_poccitanieO3 = df_poccitanie.loc[df_poccitanie['polluant'] == "O3", :]
df_poccitanieO3.head()

# Création d'un tableau avec O3 et les villes d'interêt uniquement
df_poccitanieO3_ville = df_poccitanieO3.loc[df_poccitanieO3['nom_com'].isin(
        ["MONTPELLIER", "TOULOUSE", "PERPIGNAN", "ALBI"]), :]
print(df_poccitanieO3_ville.shape)  # dimension du nouveau tableau

df_poccitanieO3_ville['nom_com'].unique()

# Histogramme du taux de 03 par villes séléctionnées
plt.figure(figsize=(8, 8))
plt.hist(df_poccitanieO3_ville['nom_com'], density=True, bins=50)
plt.xlabel("Villes")
plt.ylabel("Taux de O3")
plt.title("Taux de O3 par ville")

# Graphique en violon des taux 03 pour chaque ville choisie
sns.catplot(x="polluant", y=df_poccitanieO3_ville.columns[12], hue="nom_com", 
            data=df_poccitanieO3_ville, kind="violin", legend=False)
plt.title("Taux O3 par ville")
plt.legend(loc=1)
plt.tight_layout()

# Tableau de données pour la réalisation des ANOVA
df_poccitanieO3_ville = df_poccitanieO3_ville[['day', 'polluant', 
                                               'valeur_originale', 'nom_com']]

# Réalisation des ANOVA:
# Boxplot des taux de pollution dans chacune des villes 
# Ce boxplot nous permet d'avoir une idée sur le resultat des anova
df_poccitanieO3_ville.boxplot('valeur_originale', by='nom_com')

# ANOVA
model = ols('valeur_originale ~ nom_com', data=df_poccitanieO3_ville).fit()
aov_table = sm.stats.anova_lm(model, typ=2)  # typ=2 pour avoir un tableau 
print(aov_table)
#Random and Mixed Effects Models
##  Random Effects Models
### One-Way ANOVA
import pandas as pd
import seaborn as sb
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from pandas.api.types import CategoricalDtype
from plotnine import *
from plotnine.data import mpg
get_ipython().run_line_magic('matplotlib', 'inline')
#pip install "plotnine==0.6.0" tu use ggplot in python
###
weight = [61, 100,  56, 113,  99, 103,  75,  62,  ## sire 1
            75, 102,  95, 103,  98, 115,  98,  94,  ## sire 2
            58,  60,  60,  57,  57,  59,  54, 100,  ## sire 3
            57,  56,  67,  59,  58, 121, 101, 101,  ## sire 4
            59,  46, 120, 115, 115,  93, 105,  75 ] ## sire 5
sire=np.array([1,2,3,4,5])
sire=np.repeat(sire,8, axis=0)
animals = {'weight': weight, 'sire': pd.Categorical(sire)}
animals = pd.DataFrame(data=animals)
animals.info()
# plot
sb.stripplot(x="sire", y="weight" ,data=animals, size=10, edgecolor='red', linewidth=0.5, ax=None, dodge=True, hue="sire")
##
md = smf.mixedlm("weight ~(1-sire)", animals, groups="sire" )
mdf = md.fit()
mdf.summary()
mdf.conf_int(alpha=0.025)
from patsy.contrasts import Sum
levels = [1,2,3,4,5]
contrast = Sum().code_without_intercept(levels)
aov = ols('weight ~ C(sire, Sum)',data=animals).fit()
table = sm.stats.anova_lm(aov, typ=2) # Type 2 ANOVA DataFrame
print(table)
aov.conf_int(alpha=0.05)
randomeffect=mdf.random_effects
randomeffect
###  TA plot
x=np.array(mdf.fittedvalues)
y=np.array(mdf.resid)
#fig, ax = plt.subplots()
plt.scatter(x, y,s = 150, c = 'red', marker = '.')
plt.title('TA plot')
plt.ylabel('Pearson Residuals')
plt.xlabel('Fitted values')
plt.show()
## risiduals vs fitted plot 
Y_hat=np.random.normal(0,1,40)
Y_hat2=np.random.normal(0,1,5)
pp_x2=sm.ProbPlot(np.array([1,2,3,4,5]), fit=True)
pp_y2=sm.ProbPlot(Y_hat2, fit=True)
pp_x = sm.ProbPlot(y, fit=True)
pp_y = sm.ProbPlot(Y_hat, fit=True)
fig1= pp_x.qqplot(pp_y)
fig1.suptitle('Risiduals', fontsize=16)
fig1=plt.xlabel('theoretical quaniles')
fig2=pp_x2.qqplot(pp_y2)
fig2.suptitle('Random Effects', fontsize=16)
fig2=plt.xlabel('theoretical quaniles')
plt.xlim(-2, 2)
plt.show()
## More than one factor 
y = [142.3, 144.0, 148.6, 146.9, 142.9, 147.4, 133.8, 133.2, 
       134.9, 146.3, 145.2, 146.3, 125.9, 127.6, 108.9, 107.5,
       148.6, 156.5, 148.6, 153.1, 135.5, 138.9, 132.1, 149.7, 
       152.0, 151.4, 149.7, 152.0, 142.9, 142.3, 141.7, 141.2] 
fac=np.array([1,2,3,4])
day=np.repeat(fac,8, axis=0)
machine=np.concatenate((np.repeat(fac,2, axis=0), np.repeat(fac,2, axis=0),np.repeat(fac,2, axis=0),np.repeat(fac,2, axis=0)))
trigly = {'y': y , 'day': pd.Categorical(day), 'machine': pd.Categorical(machine)}
trigly = pd.DataFrame(data=trigly)
trigly.info()
print(pd.Categorical(day).categories)
print(pd.Categorical(machine).categories)
pd.crosstab(day, machine,rownames=['day'],colnames=['machine'])
## plot
from statsmodels.graphics.factorplots import interaction_plot
fig, ax = plt.subplots(figsize=(6, 6))
fig = interaction_plot(x=day, trace=machine, response=y,colors=['red', 'blue','brown','black'], markers=['.', '^','*','D'], ms=10, ax=ax)
## fitting model 
md2 = smf.mixedlm("y ~  (1-day)+(1-machine) + (1-day*machine) ", trigly, groups=machine)
mdf2 = md2.fit()
mdf2.summary()
print(mdf2.tvalues)
## nesting
## pastes data
Pastes=pd.read_csv('Pastes.csv',sep=" ") 
Pastes.head()
Pastes.info()
from pandas.api.types import CategoricalDtype
cask=Pastes["cask"]
batch=Pastes["batch"]
strength=Pastes["strength"]
## ggplot 
(ggplot(Pastes)+ aes(x='strength', y='cask')+ geom_point()+ labs( x='strength', y='cask')+ facet_grid('batch~'))
## model fittinh
md3= smf.mixedlm("strength ~ (batch-cask)", Pastes, groups="cask" )
mdf3 = md3.fit()
mdf3.summary()

