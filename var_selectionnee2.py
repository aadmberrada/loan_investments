# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:40:17 2021

@author: Alexis
"""

# In[271]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import vaex as vx
import numpy as np
import warnings

warnings.filterwarnings('ignore')
pd.set_option("display.max_column", 100)
pd.set_option("display.max_row", 250)


# ***Data importing***

# In[272]:


#PARQUET

pathdf = "C:/Users/Alexis/Documents/Morgane/PTYHON/projetdata.parquet"

data_parquet= pd.read_parquet(pathdf, engine='pyarrow')

#COPY DE LA BASE INITIALE
base=data_parquet.copy()

#DICTIONNAIRE
#dic=pd.read_excel("/Users/Abdoul_Aziz_Berrada/Documents/M1_EcoStat/Cours-M1/Projets/Projets_S2/loan_investments/Dictionary.xlsx")


# In[273]:


base=data_parquet.copy()


# In[274]:


print("Shape de la base de données globale:", base.shape)
#base.head()


# In[275]:


#CRÉATION DE FICO : 
base['fico'] = (base['fico_range_low']+ base['fico_range_high'])/2

#POUR NOTRE ÉTUDE ON CONSIDÉRERA LES VARAIBLES SUIVANTES : 
base = base[['loan_amnt', 'term', 'installment', 'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 
     'issue_d', 'loan_status', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal',  'revol_util', 
     'total_acc', 'application_type', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mort_acc', 'num_bc_sats', 'num_bc_tl', 
     'num_il_tl', 'num_rev_accts', 'num_sats', 'tax_liens', 'total_bc_limit', 'total_bal_ex_mort', 'earliest_cr_line', 
     'addr_state', 'fico']]


# ### SPLITTING EN TEST ET TRAIN SETS

# D'après l'énoncé, on devra utiliser comme données de test les observations de Janvier 2017 à Décembre 2018.

# In[276]:


#APERÇU DE LA BASE
print("La base de données contient",base.shape[0],"observations et", base.shape[1], "variables.\n\n")
base.head(5)


# La variable ***issue_d*** qui doit nous permettre de trancher le test_set n'est visiblement pas au bon format de date, de plus la base n'est pas ordonnée par date.
# 
# On va devoir vérifier son format et ordonner la base pour extraire le test_set

# In[277]:


base.issue_d.dtypes


# In[278]:


#CREATION DE "ISSUE_DATE" EN FORMAT DATE 
base["issue_date"]=pd.to_datetime(base["issue_d"], format='%b-%Y')

#EXTRACTION DU MOIS (NOUVELLE VARIABLE QU'ON POURRA UTLISER)
base["issue_date_month"]=base["issue_date"].dt.month

#MAINTENANT ON VA TRIER LA BASE PAR DATE(issue_date) DE LA DATE LA PLUS ANCIENNE À LA PLUS RECENTE
base=base.sort_values(by=["issue_date"], ascending=True, axis=0)

#ON SUPPRIME LES OBS AVANT 2010?
base=base[base["issue_date"]>="2010-01-01"]

base.head(5)


# In[279]:


#EST CE QU'ON A TOUS LES MOIS REPRÉSENTÉS ?
base["issue_date_month"].nunique()


# ***test_set et train_set***

# In[280]:


#TEST SET (Observations de Janvier 2017 à Décembre 2018)
test_set=base[base["issue_date"]>="2017-01-01"]
#print(test_set.head())
#print("Test_set's shape: ",test_set.shape)
#test_set.tail()


#TRAIN_SET
train_set=base[base["issue_date"]<"2017-01-01"]
#train_set.head()
#train_set.shape

print("Shape du test set",test_set.shape)
print("Shape du train set",train_set.shape)


# Dans ce qui suit, on va nous occuper de l'EDA du train_set.

# ### TRAITEMENT DU TRAIN SET

# #### TARGET 

# In[281]:


train=train_set.copy()
#CHOIX DE LA TARGET
target="loan_status"
print("Les modalités de la target dans le train set sont :\n")
print(train[target].value_counts(normalize=True))
plt.figure(figsize=(22,6))
sns.countplot(data=train, x=target)


# - D'après Investopedia (https://www.investopedia.com/terms/c/chargeoff.asp): 
# 
# A ***charge-off*** is a debt, for example on a credit card, that is deemed unlikely to be collected by the creditor because the borrower has become substantially delinquent after a period of time. However, a charge-off does not mean a write-off of the debt entirely. Having a charge-off can mean serious repercussions on your credit history and future borrowing ability. Donc il serait pertinent de mettre les clients concernés en ***Default***.
# 
# - ***In Grace Period*** et ***Late*** peuvent être mis dans les ***current***
# - ***Does not meet the credit policy. Status:Charged Off*** peut être mis dans ****charged off*** donc dans ***default***
# - ***Does not meet the credit policy. Status:Fully Paid*** peut être mis en ***Fully Paid***
# 
# 
# Donc on final, l'idée est de regrouper les modalités
# 
# - ***Does not meet the credit policy. Status:Fully Paid + Fully Paid*** : Les prêts déja ***payés*** ==> 0
# - ***Default + Does not meet the credit policy. Status:Charged Off + Charged Off*** : Les prêts en ***défaut*** ==> 1
# - ***Late (16-30 days) + Late (31-120 days) + In Grace Period + Current*** : Les prêts en ***cours*** ==>2
# 
# 

# ***Discrétisation de la target***

# In[282]:


#CHOIX DES MODALITÉS DE LA TARGET
data=train.copy()

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
print(100*train[target].value_counts(normalize=True), "\n\n")
print("La nouvelle répartition est : ")
print(100*data["loan_status"].value_counts(normalize=True), "\n\n")

#UN GRAPHIQUE POUR FAIRE BEAU
print("Visualisation des modalités de la target dans le train set")
plt.figure(figsize=(8,4))
sns.countplot(data=data, x=target)


# La répartition de la target dans le train set est très déséquélibrée. 
# 
# On procédèra sûrement à des techniques d'oversampling, d'undersampling ou de K-Fold Cross validation pour en venir à bout (Partie Modélisations)

# In[283]:


#data.head(10)


# Remarque : ***member_id***, ***mths_since_last_record*** et ***desc*** (à traiter)  ont visiblement beaucoup de NaN.
# 
# Certaines variables sont redondantes : ***grade*** et ***sub_grade***, ***loan_amount***, ***funded_amnt*** et ***funded_amnt_inv***  (à verifier)
# 
# La variable ***emp_length*** a des modalités non formalisées qu'il faudra mettre au même format pour diminuer le nombre de modalités.

# ***Export des bases de données***

# In[284]:


"""path="/Users/Abdoul_Aziz_Berrada/Documents/M1_EcoStat/Cours-M1/Projets/Projets_S2/Python_2/Database"
train= data.to_parquet(path + 'train.parquet')
test=test_set.to_parquet(path + 'test.parquet')"""


# In[285]:


#CREATION DE FONCTIONS QUI VONT PERMETTRE DE FACILITER LA PRISE EN MAIN

def definition():
    """        
    DEFINITION PERMET D'AVOIR LA DEFINITION SEULEMENT DE LA VARIABLE
    """  
    variable=input("Entrez le nom d'une variable : ")
    b=dic[dic["id"]==variable].iloc[0,1]
    print("La variable est : ", variable, "\n")
    print("Sa définition est : ", b, "\n")
    return 


def max_info():
    """        
    MAX_INFO PERMET D'AVOIR PLUS D'INFOS SUR LA VARIABLE : DEFINITION, NOMBRE DE MODALITÉS, SES MODALITÉS SI <15,
                                                  LE NOMBRE D'OBSERVATIONS PAR MODALITÉS, LE PRCTAGE DE NaN ET SON TYPE.
    """

    var=input("Entrez le nom d'une variable : ")
    a=dic[dic["id"]==var].iloc[0,1]
    print("La variable est : ", var, "\n")
    print("Sa définition est : ", a, "\n")
    c=data[var].nunique()
    d=data[var].unique()
    print("Elle a",c , "modalités différentes.\n")
    e=data[var].value_counts().sort_values()
    if c < 15: 
        print("Ses modalités sont:","\n", d , "\n")
        print(e)
    na=data[var].isna().sum()
    na_prct=100*data[var].isna().sum()/data.shape[0]
    print("La variable", var, "a au total", na, "valeurs manquantes donc {:.3f}".format(na_prct),"%."" \n")
    print("Son type est : ", data[var].dtype, "\n")
    return 


# ### VALEURS MANQUANTES

# In[286]:


nan_prct=100*data.isna().sum()/data.shape[0]
(nan_prct).sort_values(ascending=False)


# In[287]:


#VISUEL N°1
#plt.figure(figsize=(20,15))
#sns.heatmap(data.isna(), cbar=False)


# Il se passe quoi avec le Nord-Est ?

# #### Imputation des Nan

# In[288]:


#SELECTION DES VARS AVEC DES NAN
df=data[data.columns[100*data.isna().sum()/data.shape[0]>0]]
print("Il y a", df.shape[1], "variables à traiter.")
df.dtypes


# On procédera de plusieurs manières en fonction de la nature de variable cible et de la valeur de la target value.
# 
# Ainsi on distinguera les variables continues et catégorielles.
# 
# Pour faciliter l'étude, on va trier la base de données en fonction de la target, créer 2 tables, traiter les Nan, concaténer les 2 tables puis retrier la base de données en fonction de la date (comme initialement)
# 
# 

# In[289]:


target='loan_status'

    #CREATION DES 2 TABLES
d0=data[data[target]=='Fully Paid']  # PAYES
d1=data[data[target]=='Charged Off'] # EN DÉFAUT
print("Shape de d1 :", d0.shape)
print("Shape de d0 :", d1.shape)


# #### ***Variables continues***

#  **Méthodologie**
#  
# Pour les variables continues, on a décidé d'appliquer la méthodogie suivante:
# 
# 
# 1- On a avant tout filtré la base de données selon les deux modalités de la TARGET, en effet on considérera 2 cas, lorsque la target="Fully Paid" et target="Charged Off";
# 
# 2- Puis pour chaque cas, on va observer les distributions puis diagramme à moustache des variables, ainsi on imputera de la manière suivante : 
#  
# - 2.a- Si la distribution est asymétrique, avec présence potentielle d'outliers parmi les valeurs prises par la variable, on va remplacer les valeurs manquantes par la médiane;
# 
# 
# - 2.b- Si la distribution est normale, on va imputer par la moyenne.
#  
# Il est important de noter que la médiane et moyenne dont nous faisons allusion sont les médianes et moyenne de la variables en fonction de la valeur de la TARGET. Un exemple est le suivant, si lorsque la TARGET=1, la distribution d'une variable avec des Nan est symétrique, on va remplacer les Nan par la moyenne de la variable calculée dans la 'sous-base' où la TARGET=1 et non sur toute la base entière.
# 
# Notre choix d'imputer par la moyenne ou la médiane sous contraint de la modalité prise par la TARGET est motivée par une volontée de ne pas trahir ou du moins de changer le moins possible la distribution de la dite variable, en effet on est pratiquement sûr que pour une variable, remplacer les Nan pour des valeurs calculées à partir de toutes les données changeraient significativement sa distribution.
# 
# Une autre raison est que cette méthode empêche la perte de données qu'entraîne la suppression de lignes ou de colonnes.
# Cette solution n'est pas la seule qui existe et n'est sans doute pas la meilleure mais pour l'instant on s'en contentera.

# In[75]:



def visualisation():
    
    #Visualisation de la distribution, diagrammes à moustache et stats descriptives
    for col in df.select_dtypes(float):
        plt.figure(figsize=(10,8))
        plt.subplot(1,4,1)
        sns.distplot(d1[col], label='TARGET=1')
        sns.distplot(d0[col], label='TARGET=0')
        plt.title("Distribution")
        plt.legend()
        plt.subplot(1,4,2)
        sns.distplot(data[col])
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
    return

visualisation()


# À première vue, toutes ces variables ont des outliers et aucune d'elles n'a une distribution normale. On remplacera leur Nan par leur médiane.
# 

# In[86]:


for col in df.select_dtypes(float):
    print(col)


# In[290]:



liste_colonnes_num_avec_nan=["dti", "inq_last_6mths", "revol_util", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
                             
                             "mort_acc","num_bc_sats", "num_bc_tl", "num_il_tl","num_rev_accts","num_sats", 
                             
                            "total_bc_limit", "total_bal_ex_mort"]


    #ON DOIT D'ABORD STOCKER LES MEDIANES CAR ON DOIT LES UTILISER SUR LE TEST-SET
    #DONC COPIE DES BASES AVANT IMPUTATION POUR AVOIR LES MÉDIANES INITIALES
d00=d0.copy()
d11=d1.copy()


# #### ***Variables catégorielles***

# ***Méthodologie***
# 
# La méthodologie appliquée pour imputer les valeurs manquantes est la suivante:
# - Si le pourcentage de valeurs manquantes est faible (<=10%), on remplace par la valeur modale (le mode);
# - Si le pourcentage de valeurs manquantes est élevé (>10%), on créé une nouvelle modalité. Cette méthode évite la perte de données en ajoutant une catégorie.

# In[80]:


for col in df.select_dtypes(object):
    print(col)


# In[291]:


liste_colonnes_cat_avec_nan=["emp_title", "emp_length"]


# In[292]:


#emp_title
print(f"La variable emp_title a {100*d1.emp_title.isna().sum()/d1.shape[0]} % de valeurs manquantes si TARGET=1.")
print(f"La variable emp_title a {100*d0.emp_title.isna().sum()/d0.shape[0]} % de valeurs manquantes si TARGET=0.")
print("\n\n")

#emp_length

print(f"La variable emp_length a {100*d1.emp_length.isna().sum()/d1.shape[0]} % de valeurs manquantes si TARGET=1.")
print(f"La variable emp_length a {100*d0.emp_length.isna().sum()/d0.shape[0]} % de valeurs manquantes si TARGET=0.")
print("\n\n")


# Toutes les 3 variables ont des % de Nan<10% donc on remplace les Nan par les valeurs modales.

# #### ***Imputation***

# In[293]:


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


# In[294]:


data.isna().sum()


# In[112]:


data.shape


# In[110]:


#EXPORT DE LA BASE PROPRE
path="/Users/Abdoul_Aziz_Berrada/Documents/M1_EcoStat/Cours-M1/Projets/Projets_S2/Python_2/Database"
data_propre= data.to_parquet(path + 'clean_data.parquet')


# ### ***Features engineering***

# In[295]:


df = data.copy()


# In[296]:


from scipy.stats import chi2_contingency


# In[297]:


#Fonction tableau de contingence et test du khi deux, representation graphique de la stabilité des classes dans le temps
def contingence(data, VAR):
    cont = pd.crosstab(data[VAR],data["loan_status"])
    chi2, pvalue, degrees, expected = chi2_contingency(cont)
    return chi2, pvalue, degrees, expected, cont


# In[298]:


def stab(data, VAR):
    """Cette fonction extrait des tableaux pour chaque année et fait appel 
    à la fonction contingence pour calculer le tableau de contingence"""
    data2013 = data[data['issue_d'].str.contains('2013')]
    res = contingence(data2013, VAR)
    g=res[4]

    df2014 = data[data['issue_d'].str.contains('2014')]
    res = contingence(df2014, VAR)
    h=res[4]

    df2015 = data[data['issue_d'].str.contains('2015')]
    res = contingence(df2015, VAR)
    i=res[4]

    df2016 =data[data['issue_d'].str.contains('2016') ]
    res = contingence(df2016, VAR)
    j=res[4]
    
    return g,h,i,j
g,h,i,j=stab(df,'dti')


# In[299]:


#Modalité emp_title__________________________________________________________________________________

df["emp_title"]=df["emp_title"].str.lower()

def employcategorie(df):
    '''Cette fonction permet de classes les contreparties dans des catégories professionnel à partir du titre de l'emploi spécifié par ce dernier lors de la demande de prêt'''

    #Management Occupations 11
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?@]*\s?(deputy|administrator|c\.e\.o|boss|head|minister|chief|mana[a-z]*|gm|general\s|c(e|o|f)(o)?|agent|lead(er)?|direct(or|ion)|president?|executive|s?vp)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?@]*', value = r'11', regex = True, inplace=True)

    #Educational Instruction and Library Occupations 25
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(archi[a-z]*|tutor|teac?her|professor|instructor|educat(ion|or)|lecturer)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?@]*', value = r'25', regex = True, inplace=True)
    
    #Business and Financial Operations Occupations 13
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(desk|morgan|hr|mortage|treasury|decrepancy|estimator|market(er|ing)|investment|fraud|adjuster|market|affairs|public|book*eep(ing|er)|trad(er|e|es|ing)|financ[a-z]*|human\s?(ressources)|treasurer|appraiser|inspector|account(ant|ing)|cpa|cma|payroll|credit|bank(er)?|bill|tax(es)?|loan|wells\sfargo|broker|planner|front\sdesk)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'13', regex = True, inplace=True)
    
    #Sales and Related Occupations 41
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(vendor|distributor|detail[a-z]*|seller|housing|sale(s|r)?|cashier|product|demonstrator|real(tor)?|retail(er)?|buyer|merchan[a-z]*|purchas(ing|e)|wal.?mart)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'41', regex = True, inplace=True)

    #Legal Occupations 23
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(prudential|law|advocate|contract|legal|attorney|lawyer|judge|magistrate)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'23', regex = True, inplace=True)

    #Healthcare Practitioners and Technical Occupations 29
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(derma[a-z]*|cardi[a-z]*|pediatrician|cna|neuro.+|clinician|dr|ortho[a-z]*?|[a-z]*?ologist|veteri[a-z]*?|chiropractor|dentist|ct|sonograph(.r|y)|ultrasound|radio(logist)?|mri|optician|patho(logist)?|[a-z]*?grapher|para(medic)?|medic[a-z]*|pharmac|radiolog|x.?ray|therap(ist)?|health|surg(eon|ical|ery)|emt|psycha.+|lab(oratory)?|(medic|dent)al|physi(cian|cal)|doctor|optometrist|phlebo(tomist)?|n.rs(e|ing)|(l|r)(p|v)?n|crna)[&\w\s\w#\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'29', regex = True, inplace=True)

    #Computer and Mathematical Occupations 15
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(tech(nology)?|cnc|senior\sprogrammer|programmer|it|web|net(work)?|developer|analy.+|data|software|stati?sti(cian|cal)|actuary|mathemati(cian|cal)|computer|cio)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'15', regex = True, inplace=True)

    #Architecture and Engineering Occupations 17
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(cartographer|architect(or|ion)?|survey(or)?|e(n|m)gineer)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'17', regex = True, inplace=True)


    #Life, Physical, and Social Science Occupations 19
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(chemist|environment(al)?|psycho[a-z]*|scientist|economist|research(er)?|r\&d|nuclear|aero[a-z]*|chemical|physist|bio[a-z]*)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'19', regex = True, inplace=True)

    #Transportation and Material Moving Occupations 53
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(tso|air|steward|flight|boeing|air(craft|line)|transp[a-z]*|driver|truck|train|bus|chauffeur|pilot|captain|conductor|rail|taxi|port)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'53', regex = True, inplace=True)


    #Office and Administrative Support Occupations 43
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(csa|usps|registration|administration|scheduler|full\stime|salaried|superintend.nt|staff|team|specialist(s)?|ups|attendant|fedex|employee|supervisor|pack(age|er|ing)|shipping|teller|dispatch(er)?|member|delivery|shipper|letter|mail|admin(istrative)?|admistrative|cler(k|ical)|printer|postmaster|ups|secreta?ry|claims(\sadjuster)?|assistant)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'43', regex = True, inplace=True)

    #Protective Service Occupations 33
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(watcher|body\sman|federal|special\sagent|fire|investigator|custodian|patrol|police(man)|fire((\s)?fighter|man)|sheriff|arm(ed)?)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'33', regex = True, inplace=True)

    #Military Specific Occupations 55
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(marine|guard|defense|lieutenant|soldier|trooper|offi.er|navy|military|sergeant|army|usmc|sgt|usaf|major)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?@]*', value = r'55', regex = True, inplace=True)

    #Arts, Design, Entertainment, Sports, and Media Occupations 27
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(publisher|entertain(ment|er)?|translat(e|or)|musician|golf|dealer|game(s)?|pressman|casino|player|referee|act(or|ress)|theater|cast|jewel[a-z]*|audio|artist|diver|interpreter|photographer|media|desi?gner|reporter)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'27', regex = True, inplace=True)

    #Educational Instruction and Library Occupations 25
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(school|principal|book\s?[a-z]*|lecturer|teach(er|ing)|librarian|p.?rofess?or|faculty|univ[a-z]*|research)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'25', regex = True, inplace=True)

    #Food Preparation and Serving Related Occupations 35
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(pizza|butcher|meat|treat|donut|cafe(teria)?|starbucks|culinary|food(s)?|cook|dish\s?washer|chef|bak(ing|er)|meet\scutter|se?rver|bar\s?(ista|tender))[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'35', regex = True, inplace=True)

    #Construction and Extraction Occupations 47
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(restaurant|burger|mac|hvac|lineman|roof(er)?|builder|fore?man|elec[a-z]*|paint|gazier|crane|insulation|plumb(er|ing)|installer|ca.penter|mechani(c|cal|cian))[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'47', regex = True, inplace=True)

    #Community and Social Service Occupations 21
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(pta|priest|preach|planned\sparenthood|quac?ker|advisor|social|(child)?care(giver)?|counselor|community|religious|pastor|chaplain|therapist)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'21', regex = True, inplace=True)
    
    #Building and Grounds Cleaning and Maintenance Occupations 37
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(housek[a-z]*|clean(ing|er)|maid|ground|maint[a-z]*)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'37', regex = True, inplace=True)

    #Production Occupations 51
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(material|(die)?maker|sailmaker|logistic(s)?|inventory|oper.t.r|longshore(man)?|fabricator|loader|auto[a-z]*|stocker|worker|carri.r|assembler|(machine|equipment)(\soperator)?|dock|technician|machin.?st|welder|warehouse|manufactur(ing|e)|factory)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?@]*', value = r'51', regex = True, inplace=True)

    #Personal Care and Service Occupations 39 + 31 Healthcare Support Occupations
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(washer|dsp|groomer|shav.+|houseman|handyman|aide|manicurist|pct|direct\ssupport|esthetician|doorman|hha|stylist|hair|barber|nail|gambling|nann?y|funeral|crematory|train(er|ing))[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?@]*', value = r'39', regex = True, inplace=True)

    #Classe avec propriétaire d'entreprise 90
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*(associate|independant|founder|partner|self|shareholder|proprietor|owner)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'90', regex = True, inplace=True)

    #etudiant 70
    df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.;\\#\+\~\!\"\@\[\]\{|}\|\?]*(student)[&\w\s\w\s/,.\d\-\.;\(\)\':\\#\+\~\!\"\@\[\]\{|}\|\?]*', value = r'70', regex = True, inplace=True)

    #recodage des na en -1 pour le mettre dans le filtre
    df['emp_title'] = df['emp_title'].fillna("-1")
    
    #Toutes les valeurs qui ne sont ni dans les catégories sociopro ni dans les na sont recodés en -9
    #liste des valeurs autorisées
    values = ["25", "23", "13", "41", "29", "15", "17", "19", "53", "43", "33", "55", "27", "25", "35", "47", "21", "37", "51", "39", "90", "70", "11", "-1"]
    df['emp_title'].where(df['emp_title'].isin(values), other = "-9", inplace = True)

    return df


# In[300]:


#Après avoir étudier les moyennes, médianes et écarts types des salaires de chaque catégorie d'emploi, certaines d'entres elles ont été regroupées
#2 critères pour regrouper une classe : la classe etait trop petite au regarde de l'effectif totale + elle avait une moyenne/médiane/écart type proche d'une autre modalité qui elle aussi à une effectif petit
#au délà des critères de salaires etc.., la structure des catégories sociopro (en terme de diplomes) a été prise en compte : rassembler les ingénieurs et les scientifiques fait sens de mêmes que de rassembler les individus de la ventes et de l'administration (secrétaires etc..) 

def rassembler(df):
    
    df['emp_title'].replace(to_replace = "15", value = "17", inplace=True)
    df['emp_title'].replace(to_replace = "19", value = "17", inplace=True)

    df['emp_title'].replace(to_replace = "23", value = "25", inplace=True)

    df['emp_title'].replace(to_replace = "21", value =    "39", inplace=True)
    df['emp_title'].replace(to_replace = "27", value =    "39", inplace=True)
    df['emp_title'].replace(to_replace = "35", value =    "39", inplace=True)


    df['emp_title'].replace(to_replace = "37", value =    "53", inplace=True)
    df['emp_title'].replace(to_replace = "47", value =    "53", inplace=True)
    df['emp_title'].replace(to_replace = "51", value =    "53", inplace=True)

    df['emp_title'].replace(to_replace = "41", value =    "43", inplace=True)
    return df


df = employcategorie(df)
df = rassembler(df)
values = ["11","13","25","29","17"]
df['emp_title'].replace(to_replace = values, value = "Qualifiés", inplace=True)
df['emp_title'].where(df['emp_title']=="Qualifiés", other = "Non Qualifiés", inplace = True)


# In[301]:


#DISCRETISATION loan_amnt________________________________________________________________________________

df["loan_amnt"]=pd.qcut(df["loan_amnt"], 4) #Avec quartiles les modalités sont plus stable que avec déciles

#DISCRETISATION installment__________________________________________________________________________________

df["installment"]=pd.qcut(df["installment"], 3) #Avec quartiles les modalités sont plus stables que avec déciles

#Modalité home_ownership__________________________________________________________________________________
df['home_ownership'].replace(to_replace = "ANY", value = "RENT", inplace=True)
df['home_ownership'].replace(to_replace = "OTHER", value = "RENT", inplace=True)
df['home_ownership'].replace(to_replace = "NONE", value = "RENT", inplace=True)

#discretisation annual_inc________________________________________________________________________________
df["annual_inc"]=pd.qcut(df["annual_inc"], 4)

#discretisation dti ________________________________________________________________________________
df["dti"]=pd.qcut(df["dti"], 5)

#discretisation delinq_2yrs ________________________________________________________________________________
values3 = df["delinq_2yrs"]<=1
df["delinq_2yrs"]=np.where(df["delinq_2yrs"]<=1, "[0,1]" ,"]1;30]")

#Discrétisation inq_last_6mounths ________________________________________________________________________________
values3 = [8,7,6,5,4,9, 10, 11, 12]
df['inq_last_6mths'].replace(to_replace = values3, value = 3, inplace=True)

#Discrétisation open_acc ____________________________________________________________________________________________________________________________________
df["open_acc"]=pd.qcut(df["open_acc"], 3)

#discretisation pub_rec _________________________________________________________________________________________________________________________________
df["pub_rec"] = df["pub_rec"].astype('int32')
df["pub_rec"]=np.where(df["pub_rec"]<=1, "[0,1]" ,"]1;+]")

#discretisation revol_bal _________________________________________________________________________________________________________________________________
df["revol_bal"]=np.where(df["revol_bal"]<1, "[0,1[" ,"[1;+[")

#discretisation revol_util ___________________________________________________________________________________________________________________________________
df["revol_util"]=np.where(df["revol_util"]<=0, np.nan,df["revol_util"])
df["revol_util"]=pd.qcut(df["revol_util"], 2) 

#discretisation total_acc __________________________________________________________________________________________________________________________________
df["total_acc"]=pd.qcut(df["total_acc"], 5) 

#discretisation mo_sin_old_il_acct _________________________________________________________________________________
df["mo_sin_old_il_acct"]=pd.qcut(df["mo_sin_old_il_acct"],2)

#discretisation mo_sin_old_rev_tl_op ____________________________________________________________________________
df["mo_sin_old_rev_tl_op"]=pd.qcut(df["mo_sin_old_rev_tl_op"],3)

#discretisation mort_acc ____________________________________________________________________________________
values5 = df["mort_acc"]<=1
df["mort_acc"]=np.where(df["mort_acc"]<=1, "[0,1]" ,"]1;10]")

#discretisation num_bc_sats ________________________________________________________________________________
df["num_bc_sats"]=pd.qcut(df["num_bc_sats"],2)

#discretisation num_bc_tl __________________________________________________________________________________
df["num_bc_tl"]=pd.qcut(df["num_bc_tl"],2)

#discretisation num_il_tl __________________________________________________________________________________
df["num_il_tl"]=pd.qcut(df["num_il_tl"],2)

#discretisation num_rev_accts ________________________________________________________________________________
df["num_rev_accts"]=pd.qcut(df["num_rev_accts"],2)

#discretisation num_sats ________________________________________________________________________________
df["num_sats"]=pd.qcut(df["num_sats"],3)

#discretisation tax_liens ________________________________________________________________________________
df["tax_liens"]=np.where(df["tax_liens"]>=1, "[1;+[", "0")

#discretisation total_bc_limit ________________________________________________________________________________
df["total_bc_limit"]=pd.qcut(df["total_bc_limit"],3)

#discretisation total_bal_ex_mort ________________________________________________________________________________
df["total_bal_ex_mort"]=np.where(df["total_bal_ex_mort"]>=1, "[1;+[", "0")

#discretisation fico ________________________________________________________________________________
df["fico"]=pd.qcut(df["fico"], 4)


# In[302]:


df=df.drop(columns=["total_acc", "pub_rec","application_type","num_il_tl","num_rev_accts",
                    "tax_liens","total_bal_ex_mort","earliest_cr_line", "addr_state", "issue_d"], axis=1)


# In[318]:


data=df.copy()


# In[196]:


data.info()


# In[ ]:





# #### Features Selection

# ***Filter Method***

# In[200]:


import itertools
import scipy.stats as ss
#data=X.copy()
cat = data.select_dtypes(object)
cols = list(cat.columns)
corrM = np.zeros((len(cols),len(cols)))


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

# there's probably a nice pandas way to do this
for col1, col2 in itertools.combinations(cols, 2):
    idx1, idx2 = cols.index(col1), cols.index(col2)
    corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(data[col1], data[col2]))
    corrM[idx2, idx1] = corrM[idx1, idx2]

corr = pd.DataFrame(corrM, index=cols, columns=cols)
fig, ax = plt.subplots(figsize=(15, 8))
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");


# In[319]:


from sklearn.linear_model import LassoCV


# In[320]:


#TRANSFORMATION EN OBJECT
data["issue_date_month"]=data["issue_date_month"].astype(str)
data["inq_last_6mths"]=data["inq_last_6mths"].astype(str)

#BINARISATION DE LA TARGET
data.loc[data['loan_status'] == 'Fully Paid', 'loan_status'] =                                        0
data.loc[data['loan_status'] == 'Charged Off', 'loan_status'] =                                        1
data['loan_status']=data['loan_status'].astype(int)

#ENCODAGE DE LA BASE
data=pd.get_dummies(data)

#SEPARATION X ET y
X=data.drop(columns="loan_status", axis=1)
y=data["loan_status"]

#ON CHERCHE LES MEILLEURS X


# In[174]:


data.head()


# In[329]:


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC


# In[257]:


def lassoCV():
    
    print("Patience...")

        # POUR ÉVITER DE DIVISER PAR 0 EN FAISANT LE np.log10
    EPSILON = 1e-4

        #MODEL LASSOCV
    model = LassoCV(cv=5, random_state=0).fit(X, y)


    print("Meilleur alpha avec le LassoCV: %f" % model.alpha_)
    print("Meilleur score avec le LassoCV: %f" %model.score(X,y))
    coef = pd.Series(model.coef_, index = X.columns)
    print("Le Lasso a s " + str(sum(coef != 0)) + " variables et en a éliminé " +  str(sum(coef == 0)))
    imp_coef = coef.sort_values()
    
        #GRAPHIQUE
    import matplotlib
    plt.figure(figsize=(10, 15))
    imp_coef.plot(kind = "barh")
    plt.title("Importance des Variables en utilisant le Lasso")

        #RESULTATS
    plt.figure(figsize=(10, 15))
    plt.semilogx(model.alphas_ + EPSILON, model.mse_path_, ':')
    plt.plot(model.alphas_ + EPSILON, model.mse_path_.mean(axis=-1), 'k',
             label='Moyenne des MSE de tous les Folds', linewidth=2)
    plt.axvline(model.alpha_ + EPSILON, linestyle='--', color='k',
                label='alpha: CV estimé')
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel('MSE')
    plt.title('MSE sur chaque Fold')
    plt.axis('tight')
    return imp_coef

 
imp_coef= lassoCV()

# In[349]:


imp_coef[abs(imp_coef)>0.02]


# In[336]:



def lassolars():
    

    model_bic = LassoLarsIC(criterion='bic')

    model_bic.fit(X, y)

    alpha_bic_ = model_bic.alpha_

    model_aic = LassoLarsIC(criterion='aic')
    model_aic.fit(X, y)
    alpha_aic_ = model_aic.alpha_
    print("alpha AIC : ", alpha_aic_)
    print("\n")
    coef = pd.Series(model_bic.coef_, index = X.columns)
    print("Le LassoLars BIC a sélectionné " + str(sum(coef != 0)) + " variables et en a éliminé " +  str(sum(coef == 0)))
    imp_coef_bic = coef.sort_values()
    print("\n")
    print("alpha BIC : ", alpha_bic_)
    coef = pd.Series(model_aic.coef_, index = X.columns)
    print("Le LassoLars AIC a sélectionné " + str(sum(coef != 0)) + " variables et en a éliminé " +  str(sum(coef == 0)))
    imp_coef_aic = coef.sort_values()



    def plot_ic_criterion(model, name, color):
        criterion_ = model.criterion_
        plt.semilogx(model.alphas_ + EPSILON, criterion_, '--', color=color,
                     linewidth=3, label='%s criterion' % name)
        plt.axvline(model.alpha_ + EPSILON, color=color, linewidth=3,
                    label='alpha: %s estimate' % name)
        plt.xlabel(r'$\alpha$')
        plt.ylabel('criterion')


    plt.figure()
    plot_ic_criterion(model_aic, 'AIC', 'b')
    plot_ic_criterion(model_bic, 'BIC', 'r')
    plt.legend()
    plt.title("Critère d'information pour la selection du modèle")
    return
lassolars()


# In[346]:


#PRENONS LES COEFS AVEC UNE VALEUR ABSOLUE PLUS QUE 0.02
AIC= imp_coef_aic[abs(imp_coef_aic)>0.02]
BIC=imp_coef_bic[abs(imp_coef_bic)>0.02]
print("Pour l'AIC on a")
print(AIC)
print("\n")
print(BIC)


# On a exactement les mêmes variables, ce sont les mêmes qui sont choisies par le lasso.

# In[307]:

X=df.drop(columns="loan_status", axis=1)
y=df["loan_status"]


def RFE1():
    model = LinearRegression()
    #Initializing RFE model
    rfe = RFE(model, 12)
    #Transforming data using RFE
    X1 = rfe.fit_transform(X,y)  
    #Fitting the data to model
    model.fit(X1,y)
    print(rfe.support_)
    print(rfe.ranking_)
    return

RFE1()


# In[322]:

X.columns



# In[323]:

res_ref12 = [67,  4 , 5 , 6 , 7,  1 , 1, 52 ,51, 50,  1 , 1 ,60, 56 ,66, 61, 62, 63, 64, 65 ,58 ,57 ,59 ,29
 28 ,27, 49, 48 ,47 ,46 ,55, 54 ,53 ,36, 35,34 ,33 ,32,  3 , 2 ,44 ,45, 43,  1 , 1, 69, 68 , 1,
  1, 40 ,41 ,42 , 1 , 1 ,31, 30 , 1 , 1, 37 ,38 ,39 ,24 ,25,26 ,11 ,10 , 9 , 8, 23 ,21 ,20 ,16,
 22 ,18, 12, 13, 14 ,17, 15 ,19]


#liste non exhaustive : 
var_rfe12 :['term_ 36 months', 'term_ 60 months', 'emp_title_Non Qualifiés', 'emp_title_Qualifiés', 'num_bc_tl_(7.0, 70.0]', 'num_sats_(-0.001, 9.0]',
'mort_acc_[0,1]',
       'mort_acc_]1;10]']


vars_rfe=['issue_date_month_4.0','issue_date_month_5.0', 'issue_date_month_6.0', 
'issue_date_month_7.0','issue_date_month_8.0', 'issue_date_month_9.0',
'issue_date_month_2.0', 'mo_sin_old_rev_tl_op_(137.0, 198.0]',
'mo_sin_old_rev_tl_op_(198.0, 852.0]', 'mort_acc_[0,1]']


# In[ ]:





# In[ ]:





# In[350]:


#PRENDS DU TEMPS À TOURNER


# In[ ]:


def RFE_nov():  
    
    print("Prend du temps à faire tourner, donc patience...")
        #NOMBRE OPTIMAL DE VARIABLES NOV?
    nov_list=np.arange(1,13)            
    high_score=0
    nov=0           
    score_list =[]
    for n in range(len(nov_list)):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
        model = LinearRegression()
        rfe = RFE(model,nov_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score):
            high_score = score
            nov = nov_list[n]
    print("Nombre Optimal de variables: %d" %nov)
    print("Score avec %d features: %f" % (nov, high_score))
    return nov
a = RFE_nov()


# In[ ]:



cols = list(X.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, nov)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)





#On regarde le taux de défaut pas modaliuté pour chaque variable (sur df car avant one hot)
cont = pd.crosstab(df["term"],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees

cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values()



#Attention j'ai fais des modif sur la discretisation de "inq_last_6mths" qui doit avoir uniquement 3 catégories
#Les variables séléctionnées à partir des résultats du LASSO  sont les suivantes
variables_selectionnees = ['total_bc_limit_(-0.001, 10000.0]', 'total_bc_limit_(10000.0, 21300.0]',
       'total_bc_limit_(21300.0, 1105500.0]', 'home_ownership_MORTGAGE', 'home_ownership_OWN',
       'home_ownership_RENT','fico_(661.999, 672.0]',
       'fico_(672.0, 692.0]', 'fico_(692.0, 712.0]', 'fico_(712.0, 847.5]',
       'dti_(-1.001, 10.61]', 'dti_(10.61, 15.41]', 'dti_(15.41, 20.02]',
       'dti_(20.02, 25.67]', 'dti_(25.67, 999.0]', 'mort_acc_[0,1]',
       'mort_acc_]1;10]', 'annual_inc_(-0.001, 45096.0]',
       'annual_inc_(45096.0, 65000.0]', 'annual_inc_(65000.0, 90000.0]',
       'annual_inc_(90000.0, 9550000.0]', 'inq_last_6mths_0.0', 'inq_last_6mths_1.0',
       'inq_last_6mths_2.0', 'inq_last_6mths_3.0',
       'term_ 36 months', 'term_ 60 months',  'installment_(4.928999999999999, 296.82]',
       'installment_(296.82, 498.59]', 'installment_(498.59, 1584.9]']
    
    
#Pour chaque variable la modalité de référence est celle ayant le taux de défaut le plus faible (ie. le plus petit niveau de corrélation avec la cible)
modalites_references = ['total_bc_limit_(21300.0, 1105500.0]', 'home_ownership_MORTGAGE',
                        "fico_(712.0, 847.5]", 'dti_(-1.001, 10.61]', 'mort_acc_]1;10]', 
                        'annual_inc_(90000.0, 9550000.0]','installment_(4.928999999999999, 296.82]', 
                        'inq_last_6mths_0.0', 'term_ 36 months']


#à partir de data on récupère uniquement les variables selectionnees
tab = data[variables_selectionnees]

#On drop les modalités de références pour l'estimation
tab = tab.drop(columns = modalites_references)



from sklearn.linear_model import LogisticRegression
modele_logit = LogisticRegression(penalty='none',solver='newton-cg')
modele_logit.fit(tab,y)





