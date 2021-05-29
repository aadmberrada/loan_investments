import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
pd.set_option("display.max_column", 100)
pd.set_option("display.max_row", 250)

#BASE POUR TARGET=0 (FULLY-PAID)
d0= pd.read_parquet("C:\\Users\\moi\\Documents\\Scoring and loan investments\\data\\interim\\base0.parquet", engine='pyarrow')

#BASE POUR TARGET=1 (CHARGED OFF)
d1=pd.read_parquet("C:\\Users\\moi\\Documents\\Scoring and loan investments\\data\\interim\\base1.parquet", engine='pyarrow')

#TEST SET
test_set=pd.read_parquet("C:\\Users\\moi\\Documents\\Scoring and loan investments\\data\\interim\\test_set.parquet", engine='pyarrow')

target="loan_status"

#VERIFICATION DES VALEURS DE LA TARGET
print("d1 :", d1[target].value_counts())
print("d0 :", d0[target].value_counts())

print("Le test set contient", test_set.shape[0],"obs et", test_set.shape[1], "variables.")

#On crée la variable fico comme dans le train set
test_set['fico'] = (test_set['fico_range_low'] + test_set['fico_range_high'])/2 


#ON CHOISIT LES MÊMES VARIABLES QUE LE TRAIN SET (APRES FILTRAGE PAR LES IV)
vars_selected=['verification_status', 'term', 'inq_last_6mths', 'total_bal_ex_mort',
   'revol_bal', 'annual_inc', 'mort_acc', 'mo_sin_old_rev_tl_op',
   'total_bc_limit', 'dti', 'loan_amnt', 'home_ownership', 'installment',
   'mo_sin_old_il_acct', 'fico', 'revol_util', 'loan_status']

test_set=test_set[vars_selected]

#MODALITÉS DE LA TARGET
test_set[target].value_counts()

#REGROUPEMENT DES MODALITÉS DE LA TARGET (COMME DANS LE TRAIN SET)
data=test_set.copy()

#PAYÉS
data.loc[data['loan_status'] == 'Does not meet the credit policy. Status:Fully Paid', 'loan_status'] = 'Fully Paid'
data.loc[data['loan_status'] == 'Fully Paid', 'loan_status'] =                                         'Fully Paid'

#DEFAUTS
data.loc[data['loan_status'] == 'Default', 'loan_status'] =                                            'Charged Off'
data.loc[data['loan_status'] == 'Does not meet the credit policy. Status:Charged Off', 'loan_status'] ='Charged Off'
data.loc[data['loan_status'] == 'Charged Off', 'loan_status'] =                                        'Charged Off'

#EN COURS
data.loc[data['loan_status'] == 'Late (16-30 days)', 'loan_status'] =                                  'Current'
data.loc[data['loan_status'] == 'In Grace Period', 'loan_status'] =                                    'Current'
data.loc[data['loan_status'] == 'Current', 'loan_status'] =                                            'Current'
data.loc[data['loan_status'] == 'Late (31-120 days)', 'loan_status'] =                                 'Current'

#CHOIX DES MODALITÉS DE LA TARGET
data=data[data[target]!='Current']

#RÉPARTITIONS DES NOUVELLES MODALITÉS
print("L'ancienne répartition est :")
print(100*test_set[target].value_counts(normalize=True), "\n\n")
print("La nouvelle répartition est : ")
print(100*data["loan_status"].value_counts(normalize=True), "\n\n")

#GRAPHIQUE
print("Visualisation des modalités de la target dans le train set")
plt.figure(figsize=(8,4))
sns.countplot(data=data, x=target)

print("Le test set ne contient maintenant que", data.shape[0],"obs", "(au lieu de", test_set.shape[0],")", " et", 
      data.shape[1], "variables après selection des variables et des prêts courants.")

#BINARISATION DE LA TARGET____________________________________
data.loc[data['loan_status'] == 'Fully Paid', 'loan_status']  =                     0
data.loc[data['loan_status'] == 'Charged Off', 'loan_status'] =                     1
data['loan_status']=data['loan_status'].astype(int)

#TRAITEMENT DES NAN
nan_prct=100*data.isna().sum()/data.shape[0]
(nan_prct).sort_values(ascending=False)

#ON DIVISE LE TOUT EN 2 EN FONCTION DE LA VALEUR DE LA TARGET
t1=data[data[target]==1]
t0=data[data[target]==0]

#SELECTION DES VARS AVEC DES NAN
df=data[data.columns[100*data.isna().sum()/data.shape[0]>0]]
print("Il y a", df.shape[1], "variables à traiter.")
df.dtypes

col_avec_nan=["dti", "mo_sin_old_il_acct", "revol_util"]

def imputation():
    
        #IMPUTATION DES NUM
    for col in col_avec_nan:
        t0[col].fillna(d0[col].median(),inplace  = True)
        t1[col].fillna(d1[col].median(),inplace  = True)

        #CONCATÉNATION DES 2 TABLES 0 ET 1
    data=pd.concat([t0,t1])

    return data
data=imputation()

data.isna().sum()

#EXPORT DE LA BASE DE TEST PROPRE
path="C:\\Users\\moi\\Documents\\Scoring and loan investments\\data\\processed\\"
data_test= data.to_parquet(path + 'clean_test_set.parquet')












