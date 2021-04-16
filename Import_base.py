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


#df.describe()
#df["emp_title"].nunique()

#head2 = df.head(3000)
#head = head2.copy()
#head.replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(gs|manager|speech|planning|physical)[,./\s\w&\d\-\.]*', value = r'19', regex = True, inplace=True)
#head["emp_title"].iloc[1894803]
#emp_title

#df["emp_title"].iloc["zucker, goldberg, & ackerman, llc"]

#for i in range(len(df["emp_title"])):
   # if df["emp_title"].iloc[i] == "zucker, goldberg, & ackerman, llc":
    #    print(i)

#a=df.iloc[1894803]


ET = df["emp_title"]
typeemploi = ET.value_counts()
ET.nunique()


ET = df["emp_title"]
typeemploi2 = ET.value_counts()




df1=pd.read_parquet("C:/Users/morga/OneDrive/Documents/MASTER/Programmation_S2/PROJET/projetdata.parquet", engine="pyarrow")

df=df1.copy()
df["emp_title"]=df["emp_title"].str.lower()
df = df[(df['loan_status']=='Fully Paid') | (df['loan_status']=='Charged Off')]


#Educational Instruction and Library Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(archi[a-z]*|tutor|teac?her|professor|instructor|educat(ion|or)|lecturer)[&\w\s\w\s/,.\d\-\.]*', value = r'25', regex = True, inplace=True)

#Business and Financial Operations Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(desk|morgan|hr|mortage|treasury|decrepancy|estimator|market(er|ing)|investment|fraud|adjuster|market|affairs|public|book*eep(ing|er)|trad(er|e|es|ing)|financ[a-z]*|human\s?(ressources)|treasurer|appraiser|inspector|account(ant|ing)|cpa|cma|payroll|credit|bank(er)?|bill|tax(es)?|loan|wells\sfargo|broker|planner|front\sdesk)[&\w\s\w\s/,.\d\-\.]*', value = r'13', regex = True, inplace=True)

#Sales and Related Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(distributor|detail[a-z]*|seller|housing|sale(s|r)?|cashier|product|demonstrator|real(tor)?|retail(er)?|buyer|merchan[a-z]*|purchas(ing|e)|wal.?mart)[&\w\s\w\s/,.\d\-\.]*', value = r'41', regex = True, inplace=True)

#Legal Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(prudential|law|advocate|contract|legal|attorney|lawyer|judge|magistrate)[&\w\s\w\s/,.\d\-\.]*', value = r'23', regex = True, inplace=True)


#Healthcare Practitioners and Technical Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(cardi[a-z]*|pediatrician|cna|neuro.+|clinician|dr|ortho[a-z]*?|[a-z]*?ologist|veteri[a-z]*?|chiropractor|dentist|ct|sonograph(.r|y)|ultrasound|radio(logist)?|mri|optician|patho(logist)?|[a-z]*?grapher|para(medic)?|medic[a-z]*|pharmac|radiolog|x.?ray|therap(ist)?|health|surg(eon|ical|ery)|emt|psycha.+|lab(oratory)?|(medic|dent)al|physi(cian|cal)|doctor|optometrist|phlebo(tomist)?|n.rs(e|ing)|(l|r)(p|v)?n|crna)[&\w\s\w\s/,.\d\-\.]*', value = r'29', regex = True, inplace=True)


#Computer and Mathematical Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(tech(nology)?|cnc|senior\sprogrammer|programmer|it|web|net(work)?|developer|analy.+|data|software|stati?sti(cian|cal)|actuary|mathemati(cian|cal)|computer|cio)[&\w\s\w\s/,.\d\-\.]*', value = r'15', regex = True, inplace=True)


#Architecture and Engineering Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(cartographer|architect(or|ion)?|survey(or)?|e(n|m)gineer)[&\w\s\w\s/,.\d\-\.]*', value = r'17', regex = True, inplace=True)



#Life, Physical, and Social Science Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(chemist|environment(al)?|psycho[a-z]*|scientist|economist|research(er)?|r\&d|nuclear|aero[a-z]*|chemical|physist|bio[a-z]*)[&\w\s\w\s/,.\d\-\.]*', value = r'19', regex = True, inplace=True)



#Transportation and Material Moving Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(tso|air|steward|flight|boeing|air(craft|line)|transp[a-z]*|driver|truck|train|bus|chauffeur|pilot|captain|conductor|rail|taxi|port)[&\w\s\w\s/,.\d\-\.]*', value = r'53', regex = True, inplace=True)


#Office and Administrative Support Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(csa|usps|registration|administration|scheduler|full\stime|salaried|superintend.nt|staff|team|specialist(s)?|ups|attendant|fedex|employee|supervisor|pack(age|er|ing)|shipping|teller|dispatch(er)?|member|delivery|shipper|letter|mail|admin(istrative)?|cler(k|ical)|printer|postmaster|ups|secreta?ry|claims(\sadjuster)?|assistant)[&\w\s\w\s/,.\d\-\.]*', value = r'43', regex = True, inplace=True)


#Protective Service Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(watcher|body\sman|federal|special\sagent|fire|investigator|custodian|patrol|police(man)|fire((\s)?fighter|man)|sheriff|arm(ed)?)[&\w\s\w\s/,.\d\-\.]*', value = r'33', regex = True, inplace=True)

#Military Specific Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(marine|guard|defense|lieutenant|soldier|trooper|offi.er|navy|military|sergeant|army|sgt|usaf|major)[&\w\s\w\s/,.\d\-\.\(\)]*', value = r'55', regex = True, inplace=True)


#Arts, Design, Entertainment, Sports, and Media Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(entertain(ment|er)?|translat(e|or)|musician|golf|dealer|game(s)?|pressman|casino|player|referee|act(or|ress)|theater|cast|jewel[a-z]*|audio|artist|diver|interpreter|photographer|media|desi?gner|reporter)[&\w\s\w\s/,.\d\-\.]*', value = r'27', regex = True, inplace=True)


#Educational Instruction and Library Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(school|principal|book\s?[a-z]*|lecturer|teach(er|ing)|librarian|p.?rofess?or|faculty|univ[a-z]*|research)[&\w\s\w\s/,.\d\-\.\(\)]*', value = r'25', regex = True, inplace=True)

#Food Preparation and Serving Related Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(butcher|meat|treat|donut|cafe(teria)?|starbucks|culinary|food(s)?|cook|dish\s?washer|chef|bak(ing|er)|meet\scutter|se?rver|bar\s?(ista|tender))[&\w\s\w\s/,.\d\-\.]*', value = r'35', regex = True, inplace=True)

#Construction and Extraction Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(restaurant|burger|mac|hvac|lineman|roof(er)?|builder|fore?man|elec[a-z]*|paint|gazier|crane|insulation|plumb(er|ing)|installer|carpenter|mechani(c|cal|cian))[&\w\s\w\s/,.\d\-\.]*', value = r'47', regex = True, inplace=True)


#Community and Social Service Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(pta|priest|preach|planned\sparenthood|quac?ker|advisor|social|(child)?care(giver)?|counselor|community|religious|pastor|chaplain|therapist)[&\w\s\w\s/,.\d\-\.]*', value = r'21', regex = True, inplace=True)


#Building and Grounds Cleaning and Maintenance Occupations 37
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(housek[a-z]*|clean(ing|er)|maid|ground|maint[a-z]*)[&\w\s\w\s/,.\d\-\.]*', value = r'37', regex = True, inplace=True)


#Production Occupations 51 et 53
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(material|(die)?maker|sailmaker|logistic(s)?|inventory|oper.tor|longshore(man)?|fabricator|loader|auto[a-z]*|stocker|worker|carri.r|assembler|(machine|equipment)(\soperator)?|dock|technician|machin.?st|welder|warehouse|manufactur(ing|e)|factory)[\(\)&\w\s\w\s/,.\d\-\.]*', value = r'51', regex = True, inplace=True)



#Personal Care and Service Occupations 39 et 31 Healthcare Support Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(dsp|groomer|shav.+|houseman|handyman|aide|manicurist|pct|direct\ssupport|esthetician|doorman|hha|stylist|hair|barber|nail|gambling|nann?y|funeral|crematory|train(er|ing))[&\w\s\w\s/,.\d\-\.]*', value = r'39', regex = True, inplace=True)

#Target : loan_status. Charged off : company believes it will no longer collect as the borrower has become delinquent on payments.




#Classe avec propriétaire d'entreprise
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(associate|independant|founder|partner|self|shareholder|proprietor|owner)[&\w\s\w\s/,.\d\-\.]*', value = r'90', regex = True, inplace=True)


#etudiant
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*(student)[&\w\s\w\s/,.\d\-\.]*', value = r'70', regex = True, inplace=True)


#Management Occupations
df["emp_title"].replace(to_replace = r'[&\w\s\w\s/,.\d\-\.]*\s?(deputy|administrator|c\.e\.o|boss|head|minister|chief|mana[a-z]*|gm|general\s|c(e|o|f)(o)?|agent|lead(er)?|direct(or|ion)|president?|executive|s?vp)[\",./\s\w&\d\-\.\(\)]*', value = r'11', regex = True, inplace=True)



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