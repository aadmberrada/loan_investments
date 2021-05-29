"""Ce script a pour objectif de traiter le train set (valeurs manquantes, discrétisation...) avant la modélisation."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import vaex as vx
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_column", 100)
pd.set_option("display.max_row", 250)

#Entrez dans pathdf votre chemin jusqu'à la base de données de train
path_train = "C:\\Users\\moi\\Documents\\Scoring and loan investments\\data\\interim\\"

train_set= pd.read_parquet(path_train+"train_set.parquet", engine='pyarrow')

#Importation du dictionnaire qui nous sera utile par la suite :
path_dico = "C:\\Users\\moi\\Documents\\Scoring and loan investments\\references\\Dictionary.xlsx"
dic = pd.read_excel(path_dico)

#On travaille sur une copie :
train=train_set.copy()

#CREATION DE "ISSUE_DATE" EN FORMAT DATE 
train["issue_date"]=pd.to_datetime(train["issue_d"], format='%b-%Y')

#EXTRACTION DU MOIS (NOUVELLE VARIABLE QU'ON POURRA UTLISER)
train["issue_date_month"]=train["issue_date"].dt.month

#MAINTENANT ON VA TRIER LA BASE PAR DATE(issue_date) DE LA DATE LA PLUS ANCIENNE À LA PLUS RECENTE
train=train.sort_values(by=["issue_date"], ascending=True, axis=0)

#On supprimes les observations d'avant 2010 car on considère que la période, en raison de la crise, est trop particulière pour pouvoir entrainer notre modèle.
train=train[train["issue_date"]>="2010-01-01"]

#CHOIX DE LA TARGET
print("Les modalités de la target dans le train set sont :\n")
print(train['loan_status'].value_counts(normalize=True))
plt.figure(figsize=(22,6))
sns.countplot(data=train, x=target)


"""D'après Investopedia (https://www.investopedia.com/terms/c/chargeoff.asp): 

A ***charge-off*** is a debt, for example on a credit card, that is deemed unlikely to be collected by the creditor because the borrower has become substantially delinquent after a period of time. However, a charge-off does not mean a write-off of the debt entirely. Having a charge-off can mean serious repercussions on your credit history and future borrowing ability. Donc il serait pertinent de mettre les clients concernés en ***Default***.

- ***In Grace Period*** et ***Late*** peuvent être mis dans les ***current***
- ***Does not meet the credit policy. Status:Charged Off*** peut être mis dans ****charged off*** donc dans ***default***
- ***Does not meet the credit policy. Status:Fully Paid*** peut être mis en ***Fully Paid***

Donc on final, l'idée est de regrouper les modalités :

- ***Does not meet the credit policy. Status:Fully Paid + Fully Paid*** : Les prêts déja ***payés*** ==> 0
- ***Default + Does not meet the credit policy. Status:Charged Off + Charged Off*** : Les prêts en ***défaut*** ==> 1
- ***Late (16-30 days) + Late (31-120 days) + In Grace Period + Current*** : Les prêts en ***cours*** ==>2

 
"""
#On applique donc ce qu'on vient de voir :

#Les crédits remboursés
train.loc[train['loan_status'] == 'Does not meet the credit policy. Status:Fully Paid', 'loan_status'] = 'Fully Paid'
train.loc[train['loan_status'] == 'Fully Paid', 'loan_status'] =                                         'Fully Paid'

#Les défaults
train.loc[train['loan_status'] == 'Default', 'loan_status'] =                                            'Charged Off'
train.loc[train['loan_status'] == 'Does not meet the credit policy. Status:Charged Off', 'loan_status'] ='Charged Off'
train.loc[train['loan_status'] == 'Charged Off', 'loan_status'] =                                        'Charged Off'

#Les crédits en cours
train.loc[train['loan_status'] == 'Late (16-30 days)', 'loan_status'] =                                  'Current'
train.loc[train['loan_status'] == 'In Grace Period', 'loan_status'] =                                    'Current'
train.loc[train['loan_status'] == 'Current', 'loan_status'] =                                            'Current'
train.loc[train['loan_status'] == 'Late (31-120 days)', 'loan_status'] =                                 'Current'

#Nous ne pouvons pas faire de prévisions sur des crédits en cours car nous ne connaissons pas l'issue de ceux-ci. Ainsi, nous retirons des données les crédits en cours (Current).
target='loan_status'
train=train[train[target]!='Current']

#Voici la nouvele répartition des modalités :
print("La nouvelle répartition est : ")
print(100*train["loan_status"].value_counts(normalize=True), "\n\n")

print("Visualisation des modalités de la target dans le train set")
plt.figure(figsize=(8,4))
sns.countplot(data=train, x=target)

#On constate déjà un certain déséquilibre entre les deux modalités.

"""Il s'agit maintenant de traiter les variables. Nous disposons de 150 variables. Il paraît alors évident que nous ne pouvons toutes les conserver pour notre modèle. Nous réalisons ainsi plusieurs sélections :
- Nous supprimons d'abord les colonnes avec plus de 40-45% de valeurs manquantes ;
- Nous supprimons les variables dont nous ne disposerons pas au moment de l'octroi du crédit ;
- Nous supprimes les variables non pertinentes ainsi que celles ayant une définition floue."""

#Commençons par les valeurs manquantes :
pd.options.display.max_rows = 200
def na(df):
    na_total = df.isnull().sum().sort_values(ascending = False)
    na_pourcentage = df.isnull().sum().sort_values(ascending = False)/len(df)*100
    return pd.concat([na_total, na_pourcentage], axis=1, keys=['Total missing values','%'])

na(train)

list_col_na = ['sec_app_open_acc', 'sec_app_revol_util', 'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths', 'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_act_il', 'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths','sec_app_collections_12_mths_ex_med', 'sec_app_mths_since_last_major_derog', 'sec_app_fico_range_low', 'sec_app_fico_range_high', 'next_pymnt_d', 'member_id', 'revol_bal_joint', 'orig_projected_additional_accrued_interest','hardship_type', 'hardship_last_payment_amount', 'hardship_payoff_balance_amount', 'payment_plan_start_date', 'hardship_length', 'hardship_reason', 'hardship_loan_status', 'hardship_status', 'deferral_term', 'hardship_dpd', 'hardship_start_date', 'hardship_end_date', 'hardship_amount', 'dti_joint', 'annual_inc_joint','verification_status_joint', 'settlement_term', 'debt_settlement_flag_date', 'settlement_percentage', 'settlement_amount', 'settlement_status', 'settlement_date', 'desc', 'mths_since_last_record', 'il_util', 'mths_since_rcnt_il', 'all_util', 'open_acc_6m', 'total_cu_tl', 'inq_last_12m','open_rv_24m', 'open_rv_12m', 'max_bal_bc', 'total_bal_il', 'open_il_24m','open_il_12m', 'open_act_il', 'inq_fi', 'mths_since_recent_revol_delinq', 'mths_since_last_delinq']

train = train.drop(columns=list_col_na, axis=1)

#Les variables non disponibles au moment de l'octroi du crédit (1), celles non pertinentes (2) pour notre modèle ainsi que celles qui ne sont pas claires (3).

list_remove = ['acc_now_delinq',  # (1)
                     'collection_recovery_fee',  # (1)
                     'debt_settlement_flag', 
                     'funded_amnt',  # (1)
                     'funded_amnt_inv',  # (1)
                     'hardship_flag',  # (1) 
                     'initial_list_status',  # on ne sait pas ce que représentent W et F
                     'last_credit_pull_d',  # (1)
                     'last_fico_range_high',  # (1)
                     'last_fico_range_low',  # (1) 
                     'last_pymnt_d',  #(1)
                     'last_pymnt_amnt',  # (1)
                     'policy_code',  # (3)
                     'pymnt_plan',  # (1)
                     'recoveries',  # (1)
                     'out_prncp_inv',  # (1)
                     'out_prncp',  # (1)
                     'tot_hi_cred_lim',  # (3)
                     'title', #on a déjà une variable qui a les mêmes informations
                     'total_pymnt',  # (1)
                     'total_pymnt_inv',  # (1)
                     'total_rec_int',  # (1)
                     'total_rec_late_fee',  #(1)
                     'total_rec_prncp',  # (1)
                     'total_rev_hi_lim',  # (3)
                     'url',  # (2)
                     'collections_12_mths_ex_med',  # (1) 
                    'num_actv_rev_tl', # (1) 
                   'mths_since_recent_bc_dlq', # (1)
                   'total_il_high_credit_limit', # (3)
                   'bc_util', 
                  'num_actv_bc_tl', # (1)
                   'num_actv_rev_tl', 
                  'num_op_rev_tl' #(1)
                  ]

train = train.drop(columns = list_remove, axis=1)

#On crée une nouvelle variable qui nous servira. Il s'agit de la moyenne du FICO (on dispose de la borne inférieure et supérieure) au moment de l'octroi du crédit
train['fico'] = (train['fico_range_low']+ train['fico_range_high'])/2 

#On en profite pour vérifier qu'il n'y a pas de doublons dans la base de données (il faut que toutes les observations soient indépendantes)
def unique():
    un = train.nunique().sort_values(ascending=False)
    return pd.concat([un], axis=1, keys=['Number of unique values'])
unique()

#Nous allons maintenant étudier chacune des variables restantes au cas par cas pour définir celles que nous garderons pour notre étude.

#Création d'une fonction qui nous donnera directement la définition de chaque variable:
def de(col):
    b=dic[dic["id"]==col].iloc[0,1]
    print("La définition de", col, "est :", b, "\n")
    return 

de('loan_amnt')
df['loan_amnt'].describe() 

de('term')
df['term'].value_counts()

de('int_rate') 
df['int_rate'].describe()

de('installment')
df['installment'].describe()

de('grade')
df['grade'].value_counts()

de('sub_grade')

de('emp_title')
df['emp_title'].value_counts()

de('emp_length')
df['emp_length'].value_counts()

de('home_ownership')
df['home_ownership'].value_counts()

de('annual_inc')
df['annual_inc'].describe()

de('verification_status')
df['verification_status'].value_counts()

de('issue_d')
de('loan_status')

de('purpose')
df['purpose'].value_counts()

de('zip_code')

de('addr_state')
df['addr_state'].value_counts()

de('dti')
df['dti'].describe() 

de('delinq_2yrs')

de('inq_last_6mths')

de('open_acc')

de('pub_rec')

de('revol_bal')
df['revol_bal'].describe()

de('revol_util')
df['revol_util'].value_counts()

de('total_acc')
df['total_acc'].describe()

de('collection_recovery_fee')
df['collection_recovery_fee'].isnull().sum()
df['collection_recovery_fee'].describe()

de('mths_since_last_major_derog')

de('application_type')
df['application_type'].value_counts()

de('tot_coll_amt')
df['tot_coll_amt'].describe()

de('tot_cur_bal')
df['tot_cur_bal'].describe()

de('acc_open_past_24mths')

de('avg_cur_bal')

de('chargeoff_within_12_mths')

de('delinq_amnt')
df['delinq_amnt'].describe()

de('mo_sin_old_il_acct')
df['mo_sin_old_il_acct'].describe()

de('mo_sin_old_rev_tl_op')
df['mo_sin_old_rev_tl_op'].describe()

de('mo_sin_rcnt_rev_tl_op').

de('mo_sin_rcnt_tl')

de('mort_acc') 
df['mort_acc'].value_counts()

de('mths_since_recent_bc')
de('mths_since_recent_inq')

de('num_accts_ever_120_pd') 
df['num_accts_ever_120_pd'].describe() 

de('num_bc_sats') 
df['num_bc_sats'].describe() 

de('num_bc_tl')
df['num_bc_tl'].describe()

de('num_il_tl')
df['num_il_tl'].describe()

de('num_rev_accts')
df['num_rev_accts'].describe()

de('num_rev_tl_bal_gt_0') 

de('num_sats')
df['num_sats'].describe()

de('num_tl_120dpd_2m')

de('num_tl_30dpd')

de('num_tl_90g_dpd_24m')

de('num_tl_op_past_12m')

de('pct_tl_nvr_dlq')
df['pct_tl_nvr_dlq'].describe()

de('percent_bc_gt_75')
df['percent_bc_gt_75'].describe()

de('pub_rec_bankruptcies')

de('tax_liens')
df['tax_liens'].describe()

de('total_bal_ex_mort')
df['total_bal_ex_mort'].describe()

de('total_bc_limit')

de('disbursement_method')
df['disbursement_method'].value_counts()

de('debt_settlement_flag')
df['debt_settlement_flag'].value_counts()

#Finalement, voici les variables que nous décidons de conserver :

train = train[['loan_amnt', 'term', 'installment', 'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 
     'issue_d', 'loan_status', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal',  'revol_util', 
     'total_acc', 'application_type', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mort_acc', 'num_bc_sats', 'num_bc_tl', 
     'num_il_tl', 'num_rev_accts', 'num_sats', 'tax_liens', 'total_bc_limit', 'total_bal_ex_mort', 'earliest_cr_line', 
     'addr_state', 'fico', 'issue_date', 'issue_date_month']]

"""Nous allons désormais traiter les valeurs manquantes restantes. Pour la méthodologie appliquée, se référer à la partie correspondante sur document explicitant notre démarche et nos résultats. """

#Visualisation des valeurs manquantes restantes :
na(train)

#Création des deux tables :
d0=train[train[target]=='Fully Paid']  # PAYES
d1=train[train[target]=='Charged Off'] # EN DÉFAUT
print("Shape de d0 :", d0.shape)
print("Shape de d1 :", d1.shape)

for col in train.select_dtypes(float):
    print(col)
    
liste_colonnes_num_avec_nan=["dti", "inq_last_6mths", "revol_util", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
                             
                             "mort_acc","num_bc_sats", "num_bc_tl", "num_il_tl","num_rev_accts","num_sats", 
                             
                            "total_bc_limit", "total_bal_ex_mort"]
   
#Visualisation de la distribution, diagrammes à moustache et stats descriptives pour les variables quantitatives (continues)
for col in train.select_dtypes(float):
    plt.figure(figsize=(10,8))
    plt.subplot(1,4,1)
    sns.distplot(d1[col], label='TARGET=1')
    sns.distplot(d0[col], label='TARGET=0')
    plt.title("Distribution")
    plt.legend()
    plt.subplot(1,4,2)
    sns.distplot(train[col])
    plt.title("Data")
    plt.subplot(1,4,3)
    sns.boxplot(d1[col])
    plt.title("TARGET=1")
    plt.subplot(1,4,4)
    sns.boxplot(d0[col])
    plt.title("TARGET=0")
    print(d1[col].describe())
    print(col)
    print("Pourcentage de valeurs manquantes en 0 : ",100*d0[col].isna().sum()/d0.shape[0],"\n")
    print(d0[col].describe())
    print("Pourcentage de valeurs manquantes en 1 : ",100*d1[col].isna().sum()/d1.shape[0])
    print("\n\n")

#On doit d'abord stocker les médianes car on doit les utiliser sur le test set. On copie donc avant l'imputation pour avoir les médianes initiales.

d00=d0.copy()
d11=d1.copy()

#On exporte les bases :
path="C:\\Users\\moi\\Documents\\Scoring and loan investments\\data\\interim\\"
d0_= d0.to_parquet(path + 'base0.parquet')
d1_= d1.to_parquet(path + 'base1.parquet')


#Pour les variables catégorielles : 

for col in train.select_dtypes(object):
    print(col)

liste_colonnes_cat_avec_nan=["emp_title", "emp_length"]

# Toutes les variables ont des % de Nan<10% donc on remplace les Nan par les valeurs modales.

#On procède maintenant à l'imputation des valeurs manquantes :

def imputation():
    
        #IMPUTATION DES NUM
    for col in liste_colonnes_num_avec_nan:
        d00[col].fillna(d0[col].median(),inplace  = True)
        d11[col].fillna(d1[col].median(),inplace  = True)

        #IMPUTATION DES CAT
    for col in liste_colonnes_cat_avec_nan:
        d00[col].fillna(d0[col].mode()[0],inplace = True)
        d11[col].fillna(d1[col].mode()[0],inplace = True)

        #CONCATÉNATION DES 2 TABLES 0 ET 1
    data=pd.concat([d00,d11])

        #TRI DE LA TABLE EN FONCTION DE LA DATE
    data=data.sort_values(by=["issue_date"], ascending=True, axis=0)
    
        #SUPPRESION DE ISSUE_DATE
    data=data.drop(columns=["issue_date"], axis=1)
    return data
data=imputation()

na(data)

data.shape

#EXPORT DE LA BASE TRAIN PROPRE
path="C:\\Users\\moi\\Documents\\Scoring and loan investments\\data\\processed\\"
data_propre= data.to_parquet(path + 'clean_data_train.parquet')

