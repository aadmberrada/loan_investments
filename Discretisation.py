# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:14:22 2021

@author: morga
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
import re
import pyarrow as pa
from scipy.stats import chi2_contingency
import seaborn as sns

#Ouverture du DataFrame
df1=pd.read_parquet("C:/Users/morga/OneDrive/Documents/MASTER/Programmation_S2/PROJET/projetdata.parquet", engine="pyarrow")



#Première chose à faire : extraire le train avec str.contains(2017 et 2018)
masktest = (df1['issue_d'].str.contains('2017') | df1['issue_d'].str.contains('2018'))
mask_train = ~(df1['issue_d'].str.contains('2017') | df1['issue_d'].str.contains('2018'))
df = df1[mask_train]
dftest = df1[masktest]

#On ne garde que les crédits qui sont terminés (dont on est sûr de l'issue). Je ne garde vraiment que Fully Paid et Charged Off, pas Default ou les autres...

df = df[(df['loan_status']=='Fully Paid') | (df['loan_status']=='Charged Off')]


#Première suppression de variables 
list_col_na = ['sec_app_open_acc', 'sec_app_revol_util', 'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths', 'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_act_il', 'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths','sec_app_collections_12_mths_ex_med', 'sec_app_mths_since_last_major_derog', 'sec_app_fico_range_low', 'sec_app_fico_range_high', 'next_pymnt_d', 'member_id', 'revol_bal_joint', 'orig_projected_additional_accrued_interest','hardship_type', 'hardship_last_payment_amount', 'hardship_payoff_balance_amount', 'payment_plan_start_date', 'hardship_length', 'hardship_reason', 'hardship_loan_status', 'hardship_status', 'deferral_term', 'hardship_dpd', 'hardship_start_date', 'hardship_end_date', 'hardship_amount', 'dti_joint', 'annual_inc_joint','verification_status_joint', 'settlement_term', 'debt_settlement_flag_date', 'settlement_percentage', 'settlement_amount', 'settlement_status', 'settlement_date', 'desc', 'mths_since_last_record', 'il_util', 'mths_since_rcnt_il', 'all_util', 'open_acc_6m', 'total_cu_tl', 'inq_last_12m','open_rv_24m', 'open_rv_12m', 'max_bal_bc', 'total_bal_il', 'open_il_24m','open_il_12m', 'open_act_il', 'inq_fi', 'mths_since_recent_revol_delinq', 'mths_since_last_delinq']

df = df.drop(columns=list_col_na, axis=1)

#Il s'agit maintenant de réduire encore la base en supprimant des variables qui ne seront pas disponibles au moment de l'accord du crédit, celles qui dépendent directement de Lending Club telle qu'une note, celles qui ne sont pas pertinentes ou encore celles qui n'ont pas une définition claire, et donc sur lesquelles on ne peut pas compter.

#Un premier tri est alors fait :
list_remove = ['acc_now_delinq',  # won't know at time of the loan (l'info n'est pas actualisée en direct)
                     'collection_recovery_fee',  # won't know at time of the loan
                     'debt_settlement_flag', 
                     'funded_amnt',  # won't know at time of the loan 
                     'funded_amnt_inv',  # won't know at time of the loan
                     'hardship_flag',  # won't know at time of the loan + not clear 
                     'initial_list_status',  # on ne sait pas ce que représentent W et F
                     'last_credit_pull_d',  # won't know at time of loan
                     'last_fico_range_high',  # won't know at time of loan 
                     'last_fico_range_low',  # won't know at time of loan 
                     'last_pymnt_d',  #trop vieux
                     'last_pymnt_amnt',  # irrelevant + trop vieux
                     'policy_code',  # floue
                     'pymnt_plan',  # won't know at time of loan, dommage c'est une super info...
                     'recoveries',  # won't know at time of loan
                     'out_prncp_inv',  # won't know at time of loan
                     'out_prncp',  # won't know at time of loan
                     'tot_hi_cred_lim',  # flou
                     'title', #on a déjà une variable qui a les mêmes informations
                     'total_pymnt',  # won't know at time of loan ("received to date")
                     'total_pymnt_inv',  # won't know at time of loan
                     'total_rec_int',  # won't know at time of loan
                     'total_rec_late_fee',  # won't know at time of loan
                     'total_rec_prncp',  # won't know at time of loan
                     'total_rev_hi_lim',  # definition?
                     'url',  # irrelevant
                     'collections_12_mths_ex_med',  # won't know at time of application + comment ça "collections" ? 
                    'num_actv_rev_tl', #"active" 
                   'mths_since_recent_bc_dlq', #pareil
                   'total_il_high_credit_limit', #définition floue
                   'bc_util', 
                  'num_actv_bc_tl', #"active"
                  'num_actv_rev_tl', 
                  'num_op_rev_tl'] #pas l'info en temps direct et permet de supprimer une corrélation

#définition des variables avec "revolving" (rev) dans leur description : Un compte renouvelable est un compte créé par une institution financière pour permettre à un client de contracter une dette, qui est imputée au compte, et dans lequel l'emprunteur n'a pas à payer le solde impayé de ce compte en totalité chaque mois.

df = df.drop(columns = list_remove, axis=1)

#On crée une nouvelle variable qui nous servira. Il s'agit de la moyenne du FICO (on dispose de la borne inférieure et supérieure) au moment de l'octroi du crédit
df['fico'] = (df['fico_range_low']+ df['fico_range_high'])/2 



#Finalement, nous avons décidé de conserver les variables suivantes : 
df = df[['loan_amnt', 'term', 'installment', 'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'issue_d', 'loan_status', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal',  'revol_util', 'total_acc', 'application_type', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mort_acc', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_rev_accts', 'num_sats', 'tax_liens', 'total_bc_limit', 'total_bal_ex_mort', 'fico', 'earliest_cr_line', 'addr_state']]
df.shape


a=df.head(8)

#Fonction tableau de contingence et test du khi deux, representation graphique de la stabilité des classes dans le temps
def contingence(data, VAR):
    cont = pd.crosstab(data[VAR],data["loan_status"])
    chi2, pvalue, degrees, expected = chi2_contingency(cont)
    return chi2, pvalue, degrees, expected, cont

def stab(data, VAR):
    """Cette fonction extrait des tableaux pour chaque année et fait appel 
    à la fonction contingence pour calculer le tableau de contingence"""
    df2013 = data[data['issue_d'].str.contains('2013')]
    res = contingence(df2013, VAR)
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

def graph(d):
    '''Cette fonction calcul le taux de défaut par modalité'''
    d["summod"] = d["Charged Off"]+d["Fully Paid"]
    d["Taux défaut"] = d["Charged Off"]/d["summod"]
    return d["Taux défaut"]

DF4 = df.copy()

#DISCRETISATION loan_amnt________________________________________________________________________________
df["loan_amnt"].hist(density=True)
plt.show()

c=df.copy()
np.percentile(c["loan_amnt"], np.arange(0, 100, 25)) # quartiles
np.percentile(c["loan_amnt"], np.arange(0, 100, 10)) # deciles

df["loan_amnt"]=pd.qcut(df["loan_amnt"], 4) #Avec quartiles les modalités sont plus stable que avec déciles
df["loan_amnt"].value_counts    

cont = pd.crosstab(df["loan_amnt"],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees
#On rejette H0, il y a dépendance entre les classes, c'est ce qu'on cherche


#Pour la stabilité dans le temps
C3 = list(graph((stab(df,"loan_amnt"))[0])) #Taux de défaut par modalité pour 2013
C4 = list(graph((stab(df,"loan_amnt"))[1]))
C5 = list(graph((stab(df,"loan_amnt"))[2]))
C6 = list(graph((stab(df,"loan_amnt"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]], 'Classe 4': [C3[3], C4[3], C5[3], C6[3]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  
#Les classes sont stables dans le temps


#DISCRETISATION installment__________________________________________________________________________________
df["installment"].hist(density=True)
plt.show()

c=df.copy()


c["installment"]=pd.qcut(c["installment"], 3)
c["installment"].value_counts()   
c["installment"].nunique()   
d = c.head(19)

#J'ai essayé avec 10, 4 et 5 et avec 4 les tranches sont plus marquées au regard de la cible
cont = pd.crosstab(c["installment"],c["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees

#C3 = list(graph((stab(c,"installment"))[0]))
#C4 = list(graph((stab(c,"installment"))[1]))
#C5 = list(graph((stab(c,"installment"))[2]))
#C6 = list(graph((stab(c,"installment"))[3]))

#phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]], 'Classe 4': [C3[3], C4[3], C5[3], C6[3]], 'Classe 5': [C3[4], C4[4], C5[4], C6[4]], ' Classe 6': [C3[5], C4[5], C5[5], C6[5]], ' Classe 7': [C3[6], C4[6], C5[6], C6[6]], ' Classe 8': [C3[7], C4[7], C5[7], C6[7]],' Classe 9': [C3[8], C4[8], C5[8], C6[8]],' Classe 10': [C3[9], C4[9], C5[9], C6[9]]}, index = ['2013','2014', '2015', '2016'])
#plts = plt.plot(phique, marker = 'o')
#plt.legend(plts, phique.columns)  

#On rejette H0, il y a dépendance entre les classes, c'est ce qu'on cherche
cont.plot.bar()

df["installment"]=pd.qcut(df["installment"], 3) #Avec quartiles les modalités sont plus stable que avec déciles
df["installment"].value_counts()
q = df.head(65)


#Pour la stabilité dans le temps
C3 = list(graph((stab(df,"installment"))[0]))
C4 = list(graph((stab(df,"installment"))[1]))
C5 = list(graph((stab(df,"installment"))[2]))
C6 = list(graph((stab(df,"installment"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  
#Les classes se croisent --> VOIR SI ON GARDE

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

df = employcategorie(df)

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


df = rassembler(df)


values = ["11","13","25","29","17"]
df['emp_title'].replace(to_replace = values, value = "Qualifiés", inplace=True)
df['emp_title'].where(df['emp_title']=="Qualifiés", other = "Non Qualifiés", inplace = True)


ET=df['emp_title']
emploinb = ET.value_counts()


fut = df.copy()
fut.where(fut=="11", other = "-80", inplace = True)

cont = pd.crosstab(df['emp_title'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees
#On rejette H0 donc il y a dépendance entre le emp_title et le taux de défaut
#Un peu contre intuitf car les qualifiés remboursent moins que les non qualifiés 


#Pour la stabilité dans le temps
C3 = list(graph((stab(df,"emp_title"))[0]))
C4 = list(graph((stab(df,"emp_title"))[1]))
C5 = list(graph((stab(df,"emp_title"))[2]))
C6 = list(graph((stab(df,"emp_title"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  
#les modalités sont stables dans le temps

#Modalité emp_length__________________________________________________________________________________

df['emp_length'].value_counts()
cont = pd.crosstab(df['emp_length'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees

cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() #Par très explicatif cependant, on voit une différence de taux de défaut entre les <=1 et le reste


#On rejette H0 donc il y a dépendance entre le emp_title et le taux de défaut
values0= ["2 years", "3 years", "4 years", "5 years","6 years", "7 years", "8 years", "9 years", "10+ years"]
values1= ["< 1 year", "1 year"]


df['emp_length'].replace(to_replace = values0, value = "]1,+]", inplace=True)
df['emp_length'].replace(to_replace = values1, value = "]0;1]", inplace=True)


#Pour la stabilité dans le temps
C3 = list(graph((stab(df,"emp_length"))[0]))
C4 = list(graph((stab(df,"emp_length"))[1]))
C5 = list(graph((stab(df,"emp_length"))[2]))
C6 = list(graph((stab(df,"emp_length"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  
# isoler les personnes ayant un job depuis 1 an ou moins semble être la bonne décision car c'est la seule distribution pour laquelle les modalités sont satbles dans le temps (ie. les courbes de taux de défaut ne se chevauche pas)




#Modalité home_ownership__________________________________________________________________________________
df['home_ownership'].value_counts()
cont = pd.crosstab(df['home_ownership'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees

cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 
#rent à le pourcentage de défaut le plus important entre mortage et own
#On regroupe OTHER, NONE, ANY avec rent car cela 

df['home_ownership'].replace(to_replace = "ANY", value = "RENT", inplace=True)
df['home_ownership'].replace(to_replace = "OTHER", value = "RENT", inplace=True)
df['home_ownership'].replace(to_replace = "NONE", value = "RENT", inplace=True)


#Pour la stabilité dans le temps
C3 = list(graph((stab(df,"home_ownership"))[0]))
C4 = list(graph((stab(df,"home_ownership"))[1]))
C5 = list(graph((stab(df,"home_ownership"))[2]))
C6 = list(graph((stab(df,"home_ownership"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]],  'Classe 3':[C3[2], C4[2], C5[2], C6[2]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  


#discretisation annual_inc________________________________________________________________________________

df["annual_inc"].describe()

plt.hist(df["annual_inc"], range = (0, 500000), bins = 100, color = 'yellow',
            edgecolor = 'red')
plt.xlabel('valeurs')
plt.ylabel('nombres')

df["annual_inc"]=pd.qcut(df["annual_inc"], 4)
df["annual_inc"].value_counts()   
df["annual_inc"].nunique()   

cont = pd.crosstab(df['annual_inc'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees #SIgnificatif --> rejet de H0

cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 
#rent à le pourcentage de défaut le plus important entre mortage et own
#On regroupe OTHER, NONE, ANY avec rent car cela 

C3 = list(graph((stab(df,"annual_inc"))[0]))
C4 = list(graph((stab(df,"annual_inc"))[1]))
C5 = list(graph((stab(df,"annual_inc"))[2]))
C6 = list(graph((stab(df,"annual_inc"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]],  'Classe 3':[C3[2], C4[2], C5[2], C6[2]], 'Classe 4':[C3[3], C4[3], C5[3], C6[3]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  


#discretisation dti ________________________________________________________________________________

df["dti"].describe()
df["dti"].dtypes

plt.hist(df["dti"], range = (0, 40), bins = 100, color = 'yellow',
            edgecolor = 'red')


df["dti"]=pd.qcut(df["dti"], 5)
df["dti"].value_counts()   
df["dti"].nunique()   

cont = pd.crosstab(df['dti'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees #significatif, on rejette H0

cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 
#Le défaut croit avec dti


C3 = list(graph((stab(df,"dti"))[0]))
C4 = list(graph((stab(df,"dti"))[1]))
C5 = list(graph((stab(df,"dti"))[2]))
C6 = list(graph((stab(df,"dti"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 
                       'Classe 3':[C3[2], C4[2], C5[2], C6[2]], 'Classe 4': [C3[3], C4[3], C5[3], C6[3]], 
                       'Classe 5': [C3[4], C4[4], C5[4], C6[4]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns) 





#discretisation delinq_2yrs ________________________________________________________________________________
plt.hist(df["delinq_2yrs"], range = (0, 10), bins = 10, color = 'yellow',
            edgecolor = 'red')
#Deux catégories se distinguent >1 et <=1

df["delinq_2yrs"].value_counts()   
df["delinq_2yrs"].nunique() 
df["delinq_2yrs"].dtypes   
df["delinq_2yrs"] =df["delinq_2yrs"].astype('int32')


values3 = df["delinq_2yrs"]<=1
df["delinq_2yrs"]=np.where(df["delinq_2yrs"]<=1, "[0,1]" ,"]1;30]")

cont = pd.crosstab(df['delinq_2yrs'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees #significatif


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 


C3 = list(graph((stab(df,"delinq_2yrs"))[0]))
C4 = list(graph((stab(df,"delinq_2yrs"))[1]))
C5 = list(graph((stab(df,"delinq_2yrs"))[2]))
C6 = list(graph((stab(df,"delinq_2yrs"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  
#ça se croise pas c'est good




#Discrétisation inq_last_6mounths ________________________________________________________________________________
df["inq_last_6mths"].value_counts()   
df["inq_last_6mths"].nunique()   
#au vu du value count il faut au moins regrouper 8,7,6,5,4 dans 3 (t'façon si on te refuse au moins 3 prêts miskine t'es vraiment dans la merde)

values3 = [8,7,6,5,4]
df['inq_last_6mths'].replace(to_replace = values3, value = 3, inplace=True)

cont = pd.crosstab(c['inq_last_6mths'],c["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees #Significatif--> le défaut croit avec le nombre de demande de prêt dans les 6 derniers mois

cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 


C3 = list(graph((stab(df,"inq_last_6mths"))[0]))
C4 = list(graph((stab(df,"inq_last_6mths"))[1]))
C5 = list(graph((stab(df,"inq_last_6mths"))[2]))
C6 = list(graph((stab(df,"inq_last_6mths"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]], 'Classe 4': [C3[3], C4[3], C5[3], C6[3]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  
#C'est stable, c'est BG on est content les amis






#Discrétisation open_acc ____________________________________________________________________________________________________________________________________

df["open_acc"].hist(density=True)
plt.show()


df["open_acc"].value_counts()   
df["open_acc"].nunique() 
df["open_acc"].dtypes   
df["open_acc"] =df["open_acc"].astype('int32')

#Ya une valeur negative mais j'arrive pas à la virer
#df['open_acc'].replace(to_replace = c['open_acc']<0, value = np.nan, inplace=True)
#a=c[c['open_acc']<0] #COMMMMMMEEEEET C POSSIBLE JPPPPPP

df["open_acc"]=pd.qcut(df["open_acc"], 3)

cont = pd.crosstab(df['open_acc'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees 


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 
#La proportion de défauts augmente avec le nombre de compte

C3 = list(graph((stab(df,"open_acc"))[0]))
C4 = list(graph((stab(df,"open_acc"))[1]))
C5 = list(graph((stab(df,"open_acc"))[2]))
C6 = list(graph((stab(df,"open_acc"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  



#discretisation pub_rec _________________________________________________________________________________________________________________________________
plt.hist(df["pub_rec"], range = (0, 10), bins = 10, color = 'yellow',
            edgecolor = 'red')
#Quasiment exactement la même distribution que delinq_2y

df["pub_rec"].value_counts()   
df["pub_rec"].nunique() 
df["pub_rec"].dtypes   
df["pub_rec"] = c["pub_rec"].astype('int32')

df["pub_rec"]=np.where(df["pub_rec"]<=1, "[0,1]" ,"]1;+]")


cont = pd.crosstab(df['pub_rec'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #significatif --> Good


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 
#La proportion de défauts augmente avec le nombre de compte

C3 = list(graph((stab(df,"open_acc"))[0]))
C4 = list(graph((stab(df,"open_acc"))[1]))
C5 = list(graph((stab(df,"open_acc"))[2]))
C6 = list(graph((stab(df,"open_acc"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  
#ça se croise c'est nul, jpense qu'on peut la drop pcq j'ai essayé plein de truc et c'est tjr nul




#discretisation revol_bal _________________________________________________________________________________________________________________________________
plt.hist(df["revol_bal"], range = (0, 10), bins = 100, color = 'yellow',
            edgecolor = 'red')


#Quasiment exactement la même distribution que delinq_2y
df["revol_bal"].describe()
df["revol_bal"].value_counts()   
df["revol_bal"].nunique() 
df["revol_bal"].dtypes   
df["revol_bal"] = c["revol_bal"].astype('int32')

df["revol_bal"]=np.where(df["revol_bal"]<1, "[0,1[" ,"[1;+[")


cont = pd.crosstab(df['revol_bal'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #significatif 


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 


C3 = list(graph((stab(df,"revol_bal"))[0]))
C4 = list(graph((stab(df,"revol_bal"))[1]))
C5 = list(graph((stab(df,"revol_bal"))[2]))
C6 = list(graph((stab(df,"revol_bal"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  
#CONTRE TOUTES ATTENTES LES RESULTATS SONT PAS MAL




#discretisation revol_util ___________________________________________________________________________________________________________________________________

plt.hist(df["revol_util"], range = (0, 10), bins = 100, color = 'yellow',
            edgecolor = 'red')


#Quasiment exactement la même distribution que delinq_2y
df["revol_util"].describe()
df["revol_util"].value_counts()   
df["revol_util"].nunique() 
df["revol_util"].dtypes   


df["revol_util"]=np.where(df["revol_util"]<=0, np.nan,df["revol_util"])
df["revol_util"]=pd.qcut(df["revol_util"], 2) 

cont = pd.crosstab(df['revol_util'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #significatif 


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() #interessant

C3 = list(graph((stab(df,"revol_util"))[0]))
C4 = list(graph((stab(df,"revol_util"))[1]))
C5 = list(graph((stab(df,"revol_util"))[2]))
C6 = list(graph((stab(df,"revol_util"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  
#LES RESULTATS SONT PAS MAL :)



#discretisation total_acc __________________________________________________________________________________________________________________________________
plt.hist(df["total_acc"], range = (0, 10), bins = 100, color = 'yellow',
            edgecolor = 'red')


#Quasiment exactement la même distribution que delinq_2y
df["total_acc"].describe()
df["total_acc"].value_counts()   
df["total_acc"].nunique() 
df["total_acc"].dtypes   


df["total_acc"]=pd.qcut(df["total_acc"], 5) 

cont = pd.crosstab(df['total_acc'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #significatif mais bof


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() #les défaut

C3 = list(graph((stab(df,"total_acc"))[0]))
C4 = list(graph((stab(df,"total_acc"))[1]))
C5 = list(graph((stab(df,"total_acc"))[2]))
C6 = list(graph((stab(df,"total_acc"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  #LES RESULTATS SONT  NULS --> on peut drop


#Modalité application_type ________________________________________________________________________________________________________________________
df["application_type"].describe()
df["application_type"].value_counts()   
df["application_type"].nunique() 
df["application_type"].dtypes   


cont = pd.crosstab(df['application_type'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() #les défaut

C3 = list(graph((stab(df,"application_type"))[0]))
C4 = list(graph((stab(df,"application_type"))[1]))
C5 = list(graph((stab(df,"application_type"))[2]))
C6 = list(graph((stab(df,"application_type"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  
#La variable est chelou car on a pas de join app avant 2015 --> on peut drop


#discretisation mo_sin_old_il_acct _________________________________________________________________________________
plt.hist(df["mo_sin_old_il_acct"], range = (0, 10), bins = 100, color = 'yellow',
            edgecolor = 'red')


df["mo_sin_old_il_acct"].describe()
df["mo_sin_old_il_acct"].value_counts()   
df["mo_sin_old_il_acct"].nunique() 
df["mo_sin_old_il_acct"].dtypes   

df["mo_sin_old_il_acct"]=pd.qcut(df["mo_sin_old_il_acct"],2)

cont = pd.crosstab(df['mo_sin_old_il_acct'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #significatif


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() #les défaut

C3 = list(graph((stab(df,"mo_sin_old_il_acct"))[0]))
C4 = list(graph((stab(df,"mo_sin_old_il_acct"))[1]))
C5 = list(graph((stab(df,"mo_sin_old_il_acct"))[2]))
C6 = list(graph((stab(df,"mo_sin_old_il_acct"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  #GOOD




#discretisation mo_sin_old_rev_tl_op ____________________________________________________________________________

plt.hist(df["mo_sin_old_rev_tl_op"], range = (0, 10), bins = 100, color = 'yellow',
            edgecolor = 'red')


df["mo_sin_old_rev_tl_op"].describe()
df["mo_sin_old_rev_tl_op"].value_counts()   
df["mo_sin_old_rev_tl_op"].nunique() 
df["mo_sin_old_rev_tl_op"].dtypes

df["mo_sin_old_rev_tl_op"]=pd.qcut(df["mo_sin_old_rev_tl_op"],3)


cont = pd.crosstab(df['mo_sin_old_rev_tl_op'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #significatif
#il

cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() #les défaut

C3 = list(graph((stab(df,"mo_sin_old_rev_tl_op"))[0]))
C4 = list(graph((stab(df,"mo_sin_old_rev_tl_op"))[1]))
C5 = list(graph((stab(df,"mo_sin_old_rev_tl_op"))[2]))
C6 = list(graph((stab(df,"mo_sin_old_rev_tl_op"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]],'Classe 3': [C3[2], C4[2], C5[2], C6[2]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)  


#discretisation mort_acc ____________________________________________________________________________________

plt.hist(df["mort_acc"], range = (0, 10), bins = 100, color = 'yellow',
            edgecolor = 'red')


df["mort_acc"].describe()
df["mort_acc"].value_counts()   
df["mort_acc"].nunique() 
df["mort_acc"].dtypes



values5 = df["mort_acc"]<=1
df["mort_acc"]=np.where(df["mort_acc"]<=1, "[0,1]" ,"]1;10]")


cont = pd.crosstab(df['mort_acc'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #significatif!
#il

cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() #les défaut

C3 = list(graph((stab(df,"mort_acc"))[0]))
C4 = list(graph((stab(df,"mort_acc"))[1]))
C5 = list(graph((stab(df,"mort_acc"))[2]))
C6 = list(graph((stab(df,"mort_acc"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)   # NICEEE

#discretisation num_bc_sats ________________________________________________________________________________
plt.hist(df["num_bc_sats"], range = (0, 20), bins = 100, color = 'yellow',
            edgecolor = 'red')


df["num_bc_sats"].describe()
df["num_bc_sats"].value_counts()   
df["num_bc_sats"].nunique() 
df["num_bc_sats"].dtypes


df["num_bc_sats"]=pd.qcut(df["num_bc_sats"],2)


cont = pd.crosstab(df['num_bc_sats'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #significatif!
#il

cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() #les défaut

C3 = list(graph((stab(df,"num_bc_sats"))[0]))
C4 = list(graph((stab(df,"num_bc_sats"))[1]))
C5 = list(graph((stab(df,"num_bc_sats"))[2]))
C6 = list(graph((stab(df,"num_bc_sats"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)   # 2 c'est mieux



#discretisation num_bc_tl __________________________________________________________________________________
plt.hist(df["num_bc_tl"], range = (0, 20), bins = 100, color = 'yellow',
            edgecolor = 'red')


df["num_bc_tl"].describe()
df["num_bc_tl"].value_counts()   
df["num_bc_tl"].nunique() 
df["num_bc_tl"].dtypes



df["num_bc_tl"]=pd.qcut(df["num_bc_tl"],2)


cont = pd.crosstab(df['num_bc_tl'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #significatif!
#il

cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() #les défaut

C3 = list(graph((stab(df,"num_bc_tl"))[0]))
C4 = list(graph((stab(df,"num_bc_tl"))[1]))
C5 = list(graph((stab(df,"num_bc_tl"))[2]))
C6 = list(graph((stab(df,"num_bc_tl"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)   #2 c'est bof mais ça se croise pas, 3 c'est se croise , 4 c'est bordel




#discretisation num_il_tl __________________________________________________________________________________
plt.hist(df["num_il_tl"], range = (0, 20), bins = 100, color = 'yellow',
            edgecolor = 'red')


df["num_il_tl"].describe()
df["num_il_tl"].value_counts()   
df["num_il_tl"].nunique() 
df["num_il_tl"].dtypes


df["num_il_tl"]=pd.qcut(df["num_il_tl"],2)

cont = pd.crosstab(df['num_il_tl'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #significatif!
#il

cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() #les défaut

C3 = list(graph((stab(df,"num_il_tl"))[0]))
C4 = list(graph((stab(df,"num_il_tl"))[1]))
C5 = list(graph((stab(df,"num_il_tl"))[2]))
C6 = list(graph((stab(df,"num_il_tl"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns) #DROPONS TA RIEN A FAIRE C NUL




#discretisation num_rev_accts ________________________________________________________________________________
plt.hist(df["num_rev_accts"], range = (0, 20), bins = 100, color = 'yellow',
            edgecolor = 'red')

df["num_rev_accts"].describe()
df["num_rev_accts"].value_counts()   
df["num_rev_accts"].nunique() 
df["num_rev_accts"].dtypes


df["num_rev_accts"]=pd.qcut(df["num_rev_accts"],2)

cont = pd.crosstab(df['num_rev_accts'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() #les défaut

C3 = list(graph((stab(df,"num_rev_accts"))[0]))
C4 = list(graph((stab(df,"num_rev_accts"))[1]))
C5 = list(graph((stab(df,"num_rev_accts"))[2]))
C6 = list(graph((stab(df,"num_rev_accts"))[3]))

phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns) #ON DROP C NUL



#discretisation num_sats ________________________________________________________________________________
plt.hist(df["num_sats"], range = (0, 20), bins = 100, color = 'yellow',
            edgecolor = 'red')

df["num_sats"].describe()
df["num_sats"].value_counts()   
df["num_sats"].nunique() 
df["num_sats"].dtypes


df["num_sats"]=pd.qcut(df["num_sats"],3)

cont = pd.crosstab(df['num_sats'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #SIGNIFICATIF


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 

C3 = list(graph((stab(df,"num_sats"))[0]))
C4 = list(graph((stab(df,"num_sats"))[1]))
C5 = list(graph((stab(df,"num_sats"))[2]))
C6 = list(graph((stab(df,"num_sats"))[3]))


phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)#Stable dans le temps



#discretisation tax_liens ________________________________________________________________________________
plt.hist(df["tax_liens"], range = (0, 20), bins = 100, color = 'yellow',
            edgecolor = 'red')

df["tax_liens"].describe()
df["tax_liens"].value_counts()   
df["tax_liens"].nunique() 
df["tax_liens"].dtypes


df["tax_liens"]=np.where(df["tax_liens"]>=1, "[1;+[", "0")

cont = pd.crosstab(df['tax_liens'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #SIGNIFICATIF


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 

C3 = list(graph((stab(df,"tax_liens"))[0]))
C4 = list(graph((stab(df,"tax_liens"))[1]))
C5 = list(graph((stab(df,"tax_liens"))[2]))
C6 = list(graph((stab(df,"tax_liens"))[3]))


phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)#Pas stable dans le temps --> on drop



#discretisation total_bc_limit ________________________________________________________________________________
plt.hist(df["total_bc_limit"], range = (0, 20), bins = 100, color = 'yellow',
            edgecolor = 'red')

df["total_bc_limit"].describe()
df["total_bc_limit"].value_counts()   
df["total_bc_limit"].nunique() 
df["total_bc_limit"].dtypes


df["total_bc_limit"]=pd.qcut(df["total_bc_limit"],3)

cont = pd.crosstab(df['total_bc_limit'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #SIGNIFICATIF


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 

C3 = list(graph((stab(df,"total_bc_limit"))[0]))
C4 = list(graph((stab(df,"total_bc_limit"))[1]))
C5 = list(graph((stab(df,"total_bc_limit"))[2]))
C6 = list(graph((stab(df,"total_bc_limit"))[3]))


phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)


#discretisation total_bal_ex_mort ________________________________________________________________________________
plt.hist(df["total_bal_ex_mort"], range = (0, 20), bins = 100, color = 'yellow',
            edgecolor = 'red')

df["total_bal_ex_mort"].describe()
df["total_bal_ex_mort"].value_counts()   
df["total_bal_ex_mort"].nunique() 
df["total_bal_ex_mort"].dtypes


df["total_bal_ex_mort"]=np.where(df["total_bal_ex_mort"]>=1, "[1;+[", "0")

cont = pd.crosstab(df['total_bal_ex_mort'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #SIGNIFICATIF


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 

C3 = list(graph((stab(df,"total_bal_ex_mort"))[0]))
C4 = list(graph((stab(df,"total_bal_ex_mort"))[1]))
C5 = list(graph((stab(df,"total_bal_ex_mort"))[2]))
C6 = list(graph((stab(df,"total_bal_ex_mort"))[3]))


phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)#catastrophe on drop

#discretisation fico ________________________________________________________________________________
plt.hist(df["fico"], range = (0, 20), bins = 100, color = 'yellow',
            edgecolor = 'red')

df["fico"].describe()
df["fico"].value_counts()   
df["fico"].nunique() 
df["fico"].dtypes


df["fico"]=pd.qcut(df["fico"], 4)

cont = pd.crosstab(df['fico'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #SIGNIFICATIF


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 

C3 = list(graph((stab(df,"fico"))[0]))
C4 = list(graph((stab(df,"fico"))[1]))
C5 = list(graph((stab(df,"fico"))[2]))
C6 = list(graph((stab(df,"fico"))[3]))


phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]], 'Classe 4': [C3[3], C4[3], C5[3], C6[3]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)


#discretisation earliest_cr_line ________________________________________________________________________________

df=DF4.copy()
plt.hist(df["earliest_cr_line"], range = (0, 20), bins = 100, color = 'yellow',
            edgecolor = 'red')

df["earliest_cr_line"].describe()
df["earliest_cr_line"].value_counts()   
df["earliest_cr_line"].nunique() 
df["earliest_cr_line"].dtypes


df["earliest_cr_line"]=pd.qcut(df["earliest_cr_line"], 4)

cont = pd.crosstab(df['earliest_cr_line'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #SIGNIFICATIF


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 

C3 = list(graph((stab(df,"earliest_cr_line"))[0]))
C4 = list(graph((stab(df,"earliest_cr_line"))[1]))
C5 = list(graph((stab(df,"earliest_cr_line"))[2]))
C6 = list(graph((stab(df,"earliest_cr_line"))[3]))


phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]], 'Classe 4': [C3[3], C4[3], C5[3], C6[3]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)


#voir addr_state ________________________________________________________________________________
plt.hist(df["addr_state"], range = (0, 20), bins = 100, color = 'yellow',
            edgecolor = 'red')

df["addr_state"].describe()
df["addr_state"].value_counts()   
df["addr_state"].nunique() 
df["addr_state"].dtypes


df["earliest_cr_line"]=pd.qcut(df["earliest_cr_line"], 4)

cont = pd.crosstab(df['addr_state'],df["loan_status"])
chi2, pvalue, degrees, expected = chi2_contingency(cont)
chi2, pvalue, degrees  #SIGNIFICATIF


cont["summod"] = cont["Charged Off"] + cont["Fully Paid"]
cont["Taux défaut toutes années"] = cont["Charged Off"]/cont["summod"]
cont["Taux défaut toutes années"].sort_values() 

C3 = list(graph((stab(df,"addr_state"))[0]))
C4 = list(graph((stab(df,"addr_state"))[1]))
C5 = list(graph((stab(df,"addr_state"))[2]))
C6 = list(graph((stab(df,"addr_state"))[3]))


phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]], 'Classe 4': [C3[3], C4[3], C5[3], C6[3]]}, index = ['2013','2014', '2015', '2016'])
plts = plt.plot(phique, marker = 'o')
plt.legend(plts, phique.columns)


phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]]}, index = ['2013','2014', '2015', '2016'])
phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]], 'Classe 4': [C3[3], C4[3], C5[3], C6[3]]}, index = ['2013','2014', '2015', '2016'])
phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]], 'Classe 4': [C3[3], C4[3], C5[3], C6[3]],  'Classe 5': [C3[4], C4[4], C5[4], C6[4]]}, index = ['2013','2014', '2015', '2016'])
phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]], 'Classe 4': [C3[3], C4[3], C5[3], C6[3]], 'Classe 5': [C3[4], C4[4], C5[4], C6[4]], ' Classe 6': [C3[5], C4[5], C5[5], C6[5]], ' Classe 7': [C3[6], C4[6], C5[6], C6[6]], ' Classe 8': [C3[7], C4[7], C5[7], C6[7]],' Classe 9': [C3[8], C4[8], C5[8], C6[8]],'Classe 10': [C3[9], C4[9], C5[9], C6[9]]}, index = ['2013','2014', '2015', '2016'])
phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]],'Classe 4': [C3[3], C4[3], C5[3], C6[3]],'Classe 5': [C3[4], C4[4], C5[4], C6[4]]}, index = ['2013','2014', '2015', '2016'])
phique = pd.DataFrame({' Classe 1': [C3[0], C4[0], C5[0], C6[0]], 'Classe 2': [C3[1], C4[1], C5[1], C6[1]], 'Classe 3':[C3[2], C4[2], C5[2], C6[2]], 'Classe 4': [C3[3], C4[3], C5[3], C6[3]], 'Classe 5': [C3[4], C4[4], C5[4], C6[4]], ' Classe 6': [C3[5], C4[5], C5[5], C6[5]], ' Classe 7': [C3[6], C4[6], C5[6], C6[6]], ' Classe 8': [C3[7], C4[7], C5[7], C6[7]],' Classe 9': [C3[8], C4[8], C5[8], C6[8]],'Classe 10': [C3[9], C4[9], C5[9], C6[9]], 'Classe 11': [C3[10], C4[10], C5[10], C6[10]], 'Classe 12': [C3[11], C4[11], C5[11], C6[11]], 'Classe 13': [C3[12], C4[12], C5[12], C6[12]], 'Classe 14': [C3[13], C4[13], C5[13], C6[13]]}, index = ['2013','2014', '2015', '2016'])
