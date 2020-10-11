# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:24:28 2020

@author: lenovo
"""
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

