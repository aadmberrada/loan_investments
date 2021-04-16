# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:23:18 2021

@author: morga
"""
#########################
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import re
#########################

pd.options.display.max_columns = None



df1=pd.read_parquet("C:/Users/morga/OneDrive/Documents/MASTER/Programmation_S2/PROJET/projetdata.parquet", engine="pyarrow")

df=df1.copy()
df["emp_title"]=df["emp_title"].str.lower()

#Avec .parquet ça a l'air de mieux marcher (il y a pas la première colonne bizarre et le -- pour member_id)
df.shape

df.describe()
df["emp_title"].nunique()

head2 = df.head(3000)
head = head2.copy()

\w\s\w\s/

head.replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(gs|manager|speech|planning|physical)[,./\s\w&\d\-\.]*', value = r'19', regex = True, inplace=True)

head["emp_title"].iloc[17]
#emp_title



ET = df["emp_title"]
typeemploi = ET.value_counts()
ET.nunique()


ET = df["emp_title"]
typeemploi2 = ET.value_counts()

865577

df3 = df.copy()


#Management Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*\s?(mana?ge?e?(r|ment|ing)|(general\s)?superintendant|c(e|o|f)(o)?|administrat(or|ion)|lead(er)?|market(er|ing)|direct(or|ion)|president|executive|s?vp)[,./\s\w&\d\-\.]*', value = r'11', regex = True, inplace=True)


#Educational Instruction and Library Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(teac?her|professor|instructor|educat(ion|or)|lecturer)[&\w\s\w\s/,.\d\-\.]*', value = r'25', regex = True, inplace=True)


#Business and Financial Operations Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(market|affairs|public|book*eep(ing|er)|trad(er|e|es)|financ(ial|e)|human\s?(ressources)|treasurer|appraiser|inspector|account(ant|ing)|cpa|cma|payroll|credit|bank(er)?|bill|tax(es)?|loan|wells\sfargo|broker|planner|front\sdesk)[&\w\s\w\s/,.\d\-\.]*', value = r'13', regex = True, inplace=True)

#Sales and Related Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(sale(s|r)?|cashier|product|demonstrator|real(tor)?|retail(er)?|buyer|merchandiser|purchas(ing|e)|wal.?mart)[&\w\s\w\s/,.\d\-\.]*', value = r'41', regex = True, inplace=True)

#Legal Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(advocate|contract|legal|attorney|lawyer|judge|magistrate)[&\w\s\w\s/,.\d\-\.]*', value = r'23', regex = True, inplace=True)


#Healthcare Practitioners and Technical Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(veterinar(y|ian)?|chiropractor|dentist|ct|sonographer|ultrasound|radio(logist)?|mri|optician|patho(logist)?|para(medic)?|pharmac|radiolog|x.?ray|therap(ist)?|health|surg(eon|ical)|emt|psychatrist|lab(oratory)?|(medic|dent)al|physi(cian|cal)|doctor|optometrist|phlebo(tomist)?|nurs(e|ing)|(l|r)(p|v)?n|crna)[&\w\s\w\s/,.\d\-\.]*', value = r'29', regex = True, inplace=True)



#Computer and Mathematical Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(technology|cnc|senior\sprogrammer|programmer|it|web|net(work)?|developer|analyst|data|software|stati?sti(cian|cal)|actuary|mathemati(cian|cal)|computer|cio)[&\w\s\w\s/,.\d\-\.]*', value = r'15', regex = True, inplace=True)



#Architecture and Engineering Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(architect(or|ion)?|survey(or)?|engineer)[&\w\s\w\s/,.\d\-\.]*', value = r'17', regex = True, inplace=True)



#Life, Physical, and Social Science Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(chemist|environment(al)?|psychologist|scientist|economist|research(er)?|nuclear|chemical|physist|bio[a-z]*)[&\w\s\w\s/,.\d\-\.]*', value = r'19', regex = True, inplace=True)



#Transportation and Material Moving Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(scheduler|flight|boeig|aircraft|transport(ation)?|driver|train|bus|pilot|captain|conductor|rail|taxi)[&\w\s\w\s/,.\d\-\.]*', value = r'53', regex = True, inplace=True)


#Office and Administrative Support Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(fedex|employee|supervisor|package|shipping|teller|dispatch(er)?|letter|mail|admin(istrative)?|cler(k|ical)|printer|postmaster|ups|secretary|claims(\sadjuster)?|assistant)[&\w\s\w\s/,.\d\-\.]*', value = r'43', regex = True, inplace=True)



#Protective Service Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(special\sagent|fire|investigator|custodian|patrol|police(man)|fire((\s)?fighter|man)|sheriff|arm(ed)?)[&\w\s\w\s/,.\d\-\.]*', value = r'33', regex = True, inplace=True)

#Military Specific Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(defense|lieutenant|soldier|trooper|officer|navy|military|sergeant)[&\w\s\w\s/,.\d\-\.]*', value = r'55', regex = True, inplace=True)


#Arts, Design, Entertainment, Sports, and Media Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(audio|artist|interpreter|photographer|media|desi?gner|reporter)[&\w\s\w\s/,.\d\-\.]*', value = r'27', regex = True, inplace=True)


#Educational Instruction and Library Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(school|principal|lecturer|teach(er|ing)|librarian|professor|faculty|university|research)[&\w\s\w\s/,.\d\-\.]*', value = r'25', regex = True, inplace=True)

#Food Preparation and Serving Related Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(cafeteria|starbucks|culinary|food(s)?|cook|dishwasher|chef|bak(ing|er)|meet\scutter|se?rver|bar\s?(ista|tender))[&\w\s\w\s/,.\d\-\.]*', value = r'35', regex = True, inplace=True)

#Construction and Extraction Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(hvac|lineman|builder|fore?man|electr(ici?an|onics?)|paint|gazier|crane|insulation|plumber|installer|carpenter|mechani(c|cal|cian))[&\w\s\w\s/,.\d\-\.]*', value = r'47', regex = True, inplace=True)


#Community and Social Service Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(advisor|social|(child)?care(giver)?|counselor|community|religious|pastor|chaplain|therapist)[&\w\s\w\s/,.\d\-\.]*', value = r'21', regex = True, inplace=True)


#Building and Grounds Cleaning and Maintenance Occupations 37
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(housekeep(ing|er)|clean(ing|er)|maid|ground|maintenance)[&\w\s\w\s/,.\d\-\.]*', value = r'37', regex = True, inplace=True)


#Production Occupations 51 et 53
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(inventory|operator|longshore(man)?|fabricator|loader|auto[a-z]*|stocker|worker|carrier|assembler|(machine|equipment)(\soperator)?|dock|technician|machin.?st|welder|warehouse|manufactur(ing|e)|factory)[&\w\s\w\s/,.\d\-\.]*', value = r'51', regex = True, inplace=True)



#Personal Care and Service Occupations 39 et 31 Healthcare Support Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(aide|manicurist|pct|direct\ssupport|esthetician|doorman|hha|stylist|hair|barber|nail|gambling|nanny|funeral|crematory|train(er|ing))[&\w\s\w\s/,.\d\-\.]*', value = r'39', regex = True, inplace=True)

#Target : loan_status. Charged off : company believes it will no longer collect as the borrower has become delinquent on payments.


#Classe avec associate et adjoint
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(deputy|associate)[&\w\s\w\s/,.\d\-\.]*', value = r'80', regex = True, inplace=True)

#Classe avec propriétaire d'entreprise
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(partner|self|shareholder|proprietor|owner)[&\w\s\w\s/,.\d\-\.]*', value = r'90', regex = True, inplace=True)









"""
X.loc[X['loan_status'] == 'Current', 'loan_status'] = 1
X.loc[X['loan_status'] == 'Fully Paid', 'loan_status'] = 1
X.loc[X['loan_status'] == 'In Grace Period', 'loan_status'] = 1
X.loc[X['loan_status'] == 'Charged Off', 'loan_status'] = 0
X.loc[X['loan_status'] == 'Late (31-120 days)', 'loan_status'] = 0
X.loc[X['loan_status'] == 'Late (16-30 days)', 'loan_status'] = 0
X.loc[X['loan_status'] == 'Default', 'loan_status'] = 0"""
# => on pourrait mettre 1 pour les crédits à risque élevé ou déjà en défaut et 0 
#pour les bons crédits
#On peut aussi seulement garder les crédits Fully Paid et Charged 
#Off (Default aussi). Ce sont les crédits qui sont terminés, et donc on sait avec certitude si le client a remboursé ou pas.