# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 21:27:15 2021

@author: morga
"""

#########################
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import re
import numpy as np
import matplotlib.pyplot as plt
#########################

pd.options.display.max_columns = None

df1=pd.read_parquet("C:/Users/morga/OneDrive/Documents/MASTER/Programmation_S2/PROJET/projetdata.parquet", engine="pyarrow")

df=df1.copy()
df["emp_title"]=df["emp_title"].str.lower()
df = df[(df['loan_status']=='Fully Paid') | (df['loan_status']=='Charged Off')]
mask_year = (df['issue_d'].str.contains('2011')|df['issue_d'].str.contains('2012')|df['issue_d'].str.contains('2013')|df['issue_d'].str.contains('2014')|df['issue_d'].str.contains('2015')|df['issue_d'].str.contains('2016')|df['issue_d'].str.contains('2017')|df['issue_d'].str.contains('2018'))
df = df[mask_year]

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

ET=df['emp_title']
emploinb = ET.value_counts()
    
