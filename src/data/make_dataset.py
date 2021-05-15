# -*- coding: utf-8 -*-
import pandas as pd

#Entrez dans pathdf votre chemin jusqu'à la base de données brutes. Veillez à bien extrairer la base data.parquet avant l'importation, car elle est trop volumineuse.
pathdf = "C:\\Users\\moi\\Documents\\Scoring and loan investments\\data\\raw\\"

data_parquet= pd.read_parquet(pathdf+"data.parquet", engine='pyarrow')

#On crée une copie du dataframe pour travailler dessus :
base=data_parquet.copy()

#Taille de la base de données : 
print("Shape de la base de données globale:", base.shape)

#Commençons par séparer notre base en test et train set. Il y a potentiellement certaines opérations que nous appliquerons sur le base de train mais pas sur la test. Nous considérons les crédits à partir du 1er janvier 2017 et jusqu'à fin 2018 comme notre base de test.

mask_test = (base['issue_d'].str.contains('2017') | base['issue_d'].str.contains('2018'))
mask_train = ~(base['issue_d'].str.contains('2017') | base['issue_d'].str.contains('2018'))
test_set = base[mask_test]
train_set = base[mask_train]

#Exportons le test set :
path_test="C:\\Users\\moi\\Documents\\Scoring and loan investments\\data\\interim\\"
data_test= test_set.to_parquet(path_test + 'test_set.parquet')


#Puis le train set : 
path_train="C:\\Users\\moi\\Documents\\Scoring and loan investments\\data\\interim\\"
data_train= train_set.to_parquet(path_train + 'train_set.parquet')
