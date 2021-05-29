import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scorecardpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, classification_report, auc, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings('ignore')
pd.set_option("display.max_column", 100)
pd.set_option("display.max_row", 250)


#Importation des données

base1= pd.read_parquet("C:\\Users\\moi\\Documents\\Scoring and loan investments\\data\\processed\\clean_data_train.parquet", engine='pyarrow')

df = base1.copy()

#Avant de continuer la modélisation, il nous faut traiter la variable emp_title de façon particulière :

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


dff=df.copy()

#On décide de supprimer encore des colonnes avec trop de modalités différentes
dff=dff.drop(columns=["earliest_cr_line", "addr_state", 'issue_d'], axis=1)

data=dff.copy()

data.info()

#TRANSFORMATION EN OBJECT
data["issue_date_month"]=data["issue_date_month"].astype(str)

#BINARISATION DE LA TARGET
data.loc[data['loan_status'] == 'Fully Paid', 'loan_status']  =                     0
data.loc[data['loan_status'] == 'Charged Off', 'loan_status'] =                     1
data['loan_status']=data['loan_status'].astype(int)

#SEPARATION X ET y
X=data.drop(columns="loan_status", axis=1)
y=data["loan_status"]

#Discrétisation des variables__________________________________________

def creation_des_classes():    #CREATION DES CLASSES PAR "CHIMERGE"
    classes = sc.woebin(data, y="loan_status", positive=1, method="chimerge")
    return classes

classes=creation_des_classes()

#GRAPHIQUE DES CLASSES
sc.woebin_plot(classes)

def filtre_base(iv):
    #FILTRE DE LA BDD
    print("Cette opération peut prendre quelques minutes...\n")
    dat=sc.var_filter(data, y="loan_status",iv_limit=iv)
    print("\n")
    print("Après filtrage de la base en gardant les variables avec une IV>", iv, "les variables sélectionnées sont :\n", dat.columns)
    return dat

dat=filtre_base(0.02)

#SPLIT DE LA BASE EN TRAIN ET VALID SETS EN 75/25
def split(train_size):
    train_set, valid_set = sc.split_df(dat, y="loan_status", ratio=train_size, seed=186).values()
    print('Le train_size est :', train_size)
    
    print("\n")
    print('Shape train_set', train_set.shape)
    print('Shape valid_set', valid_set.shape)
    return train_set, valid_set

train_set, valid_set = split(0.75)

#CONVERSION DES CLASSES DES VARIABLES DES TRAIN ET VALID SETS EN WOE
def conversion_woe():
    print("Cette opération peut prendre quelques minutes...\n")
    train_woe = sc.woebin_ply(train_set, classes)
    valid_woe = sc.woebin_ply(valid_set, classes)
    
    #CREATION DES X ET y DES TRAIN ET VALID SETS
    y_train = train_woe.loc[:,"loan_status"]
    X_train = train_woe.loc[:,train_woe.columns != "loan_status"]
    y_valid  = valid_woe.loc[:,"loan_status"]
    X_valid  = valid_woe.loc[:,train_woe.columns != "loan_status"]
    
    return y_train, X_train, y_valid, X_valid, train_woe, valid_woe

y_train, X_train, y_valid, X_valid, train_woe, valid_woe=conversion_woe()

#Regression logistique____________________________

def modèle_1():

    m1 = LogisticRegression(penalty='l1', C=5, solver='saga',max_iter= 100, random_state=0).fit(X_train, y_train)
    print("Le modèle 1 est:", m1, "\n")
    # m1.coef_
    # m1.intercept_

        #PREDICTION DES Y_PRED
    train_pred_m1 = m1.predict_proba(X_train)[:,1]
    valid_pred_m1 = m1.predict_proba(X_valid)[:,1]
    
        #SCORES
    train_score_m1=m1.score(X_train, y_train)
    print("train score : ",train_score_m1)
    
    valid_score_m1=m1.score(X_valid, y_valid)
    print("valid score : ",valid_score_m1)

        #AUC
    fpr_m1, tpr_m1, _=roc_curve(y_valid, valid_pred_m1)
    roc_auc_m1=auc(fpr_m1, tpr_m1)
    print("L'AUC est de", roc_auc_m1)
    print("\n")
        #ACCURACY, PRECISION... AU SEUIL DE 50% DE CONFIANCE (STANDARD)
    print("classification_report\n")
    valid_pred_m1_b=valid_pred_m1 >0.5
    class_report_m1=classification_report(y_valid, valid_pred_m1_b)
    print(class_report_m1)
    print("\n")

        #MATRICE DE CONFUSION
    conf_mat_m1=confusion_matrix(y_valid, valid_pred_m1_b)
    print("confusion_matrix\n")
    print(conf_mat_m1)
    return m1, m1.coef_, roc_auc_m1, class_report_m1, conf_mat_m1, train_score_m1, valid_score_m1

m1, m1.coef_, roc_auc_m1, class_report_m1, conf_mat_m1, train_score_m1, valid_score_m1=modèle_1()

#Un peu de GridSearch pour obtenir un meilleur modèle ?
relog=LogisticRegression(random_state=0, solver='saga')
param={"penalty":["l2", "l1", "elasticnet"], 'C':[5,10,15,20], "max_iter":[100,150,200]}

relog_acc=GridSearchCV(relog, param_grid=param).fit(X_train, y_train)
print('LogisticRegression')
    #OPTIMISATION DE L'ACCURACY (MÉME SI PAS TROP DETERMINANT)
print(" - OPTIMISATION DE L'ACCURACY")
print("Les meilleurs paramètres sont:", relog_acc.best_params_)
print("Le meilleur score est:", relog_acc.best_score_)

print('LogisticRegression')
relog_auc=GridSearchCV(relog, param_grid=param, scoring='roc_auc').fit(X_train, y_train)
    #OPTIMISATION DE L'AUC
print(" - OPTIMISATION DE L'AUC")
print("Les meilleurs paramètres sont:", relog_auc.best_params_)
print("Le meilleur score est:", relog_auc.best_score_)

#Une autre en optimisant le recall :
relog=LogisticRegression()
param={"penalty":["l2", "l1"], 'C':[5,10,15,20], "max_iter":[25,100,150,200], 'solver':['lbfgs','saga']}
relog_rec=GridSearchCV(relog, param_grid=param, scoring='recall').fit(X_train, y_train)
print('LogisticRegression')
print(" - OPTIMISATION DU RECALL")
print("Les meilleurs paramètres sont:", relog_rec.best_params_)
print("Le meilleur score est:", relog_rec.best_score_)

#On essaie de faire de la cross validation :

def validation_croisée(KFold):
    
    print('On fait une cross validation avec :', KFold, "Folds")
    print("Le modèle utilisé est : ", m1)
        #ON PREND LE MEILLEUR MODÈLE POUR L'INSTANT
    clf = m1

        #ON APPLIQUE LES WOE A TOUTE LA BASE POUR APRÈS PRENDRE TOUTES LES FEATURES
    dataa=sc.woebin_ply(dat, classes)

        #ON RECUP X_ ET y_ JUSTE POUR LA VC
    X_=dataa.drop(columns=["loan_status"], axis=1)
    y_=dataa["loan_status"]

        #CROSS VAL AVEC 5 FOLDS
    scores=cross_val_score(clf, X_, y_, cv=KFold, scoring='roc_auc')
    print("L'AUC de la CV est de :", "%0.2f + ou - %0.2f" % (scores.mean(), scores.std()))

        #PREDICTION POUR VOIR LES METRIQUES
    y_pred = cross_val_predict(clf, X_, y_, cv=KFold, method='predict_proba')[:,1]
    y_pred_b =y_pred>0.5

        #METRIQUES
            #AUC
    fpr_vc, tpr_vc, _=roc_curve(y_, y_pred)
    roc_auc_vc=auc(fpr_vc, tpr_vc)
    #print("L'AUC de la CV est de", roc_auc_vc)
    print("L'AUC du modèle m1 est de :", roc_auc_m1)
    print("\n")
            #CLASS REPORT
    class_report_cv=classification_report(y_, y_pred_b)
    print("Classification report vc : \n")
    print(class_report_cv)
    print("\n")
    print("Classification report m1 : \n")
    print(class_report_m1)
    return 

validation_croisée(5)

#Enfin, nous essayons une dernière technique pour améliorer nos résultats : l'oversampling

def over_sampling():
    
    #ON VA FAIRE DE MANIÈRE AUTO
    print("On va rééchantillonner le train set...\n")
    from imblearn.over_sampling import SMOTE
    
    smote = SMOTE(random_state=0 ,k_neighbors=15)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(smote.get_params(deep=True))
    print("\n")

    from collections import Counter
    print("La nouvelle proportion de 0 et 1 de la target resamplée est: %s" % Counter(y_train_smote))
    print("\n")
    print(y_train_smote.value_counts(normalize=True))
    
    return X_train_smote, y_train_smote

X_train_smote, y_train_smote = over_sampling()

#On revient à la régression logistique après le SMOTE

def modèle_2():
    
    m2=LogisticRegression(penalty='l2', C=5, solver='saga', max_iter=100, random_state=0).fit(X_train_smote, y_train_smote)
    
    print("La modèle m2 est :", m2)
    print("m2.fit(X_train_smote, y_train_smote) \n")
    
        #PREDICTIONS À PARTIR DU SMOTE
    valid_pred_m2=m2.predict_proba(X_valid)[:,1]
    train_pred_m2=m2.predict_proba(X_train_smote)[:,1]

        #SCORES DU 2E MODÈLE RESAMPLÉ
    train_score_m2=m2.score(X_train_smote, y_train_smote)
    valid_score_m2=m2.score(X_valid, y_valid)
    
    print("train score M2 :", train_score_m2)
    print("valid score M2 :", valid_score_m2)
    print("train score M1 :", train_score_m1)
    print("valid score M1 :", valid_score_m1)
    
        #AUC
    print("\n")
    tpr_m2, fpr_m2, _=roc_curve(y_valid, valid_pred_m2)
    roc_auc_m2=auc(tpr_m2, fpr_m2)
    print("AUC M2 : ",roc_auc_m2)
    print("AUC M1 : ", roc_auc_m1)

        #CONFUSION MATRIX
    print("\n")
    valid_pred_m2_b=valid_pred_m2>0.5
    conf_mat_m2=confusion_matrix(y_valid, valid_pred_m2_b)
    print('Matrice de confusion M2 :\n', conf_mat_m2)
    print("Matrice de confusion M1 :\n", conf_mat_m1)
    
        #CLASSIFICATION REPORT
    print("\n")
    class_report_m2= classification_report(y_valid, valid_pred_m2_b)
    print("classification_report M2 :\n",class_report_m2)
    print("classification_report M1 :\n",class_report_m1)
    
    return m2, m2.coef_, roc_auc_m2, class_report_m2, conf_mat_m2, train_score_m2, valid_score_m2

m2, m2.coef_, roc_auc_m2, class_report_m2, conf_mat_m2, train_score_m2, valid_score_m2=modèle_2()

#Nouvelle GridSearch 

dta=LogisticRegression()
param={"penalty":["l2", "l1"], 'C':[5,10,15,20], "max_iter":[100,150,200]}
dta_auc=GridSearchCV(dtt, param_grid=param, scoring='roc_auc').fit(X_train_smote, y_train_smote)
    #OPTIMISATION DE L'AUC
print(" - OPTIMISATION DE L'AUC")
print("Les meilleurs paramètres sont:", dta_auc.best_params_)
print("Le meilleur score est:", dta_auc.best_score_)

dtr=LogisticRegression()
param={"penalty":["l2", "l1"], 'C':[5,10,15,20], "max_iter":[100,150,200]}
dtr_rec=GridSearchCV(dtt, param_grid=param, scoring='recall').fit(X_train_smote, y_train_smote)
    #OPTIMISATION DE RECALL
print(" - OPTIMISATION DU RECALL")
print("Les meilleurs paramètres sont:", dtr_rec.best_params_)
print("Le meilleur score est:", dtr_rec.best_score_)

dtf=LogisticRegression()
param={"penalty":["l2", "l1"], 'C':[5,10,15,20], "max_iter":[100,150,200]}
dtf_f=GridSearchCV(dtt, param_grid=param, scoring='f1').fit(X_train_smote, y_train_smote)
    #OPTIMISATION DU F1-SCORE
print(" - OPTIMISATION DU F1-SCORE")
print("Les meilleurs paramètres sont:", dtf_f.best_params_)
print("Le meilleur score est:", dtf_f.best_score_)


#Application au test set_____________________________________________

base2= pd.read_parquet("C:\\Users\\moi\\Documents\\Scoring and loan investments\\data\\processed\\clean_test_set.parquet", engine='pyarrow')
test_set = base2.copy()

def test_woe():
    test_set=base2.copy()

        #CONVERSION DES CLASSES DES VARIABLES DES TRAIN ET TEST SETS EN WOE
    test_woe = sc.woebin_ply(test_set, classes)

        #CREATION DES X ET y DES TRAIN ET TEST SETS
    y_test  = test_woe.loc[:,"loan_status"]
    X_test  = test_woe.loc[:,train_woe.columns != "loan_status"]
    
    return y_test, X_test

y_test, X_test=test_woe()

def modèle():
    #LE MODÈLE FINAL QUI SERA APPLIQUÉ AU TEST SET 
    print("Voici notre meilleur modèle \n")
    m=LogisticRegression(penalty='l2', C=5, solver='saga', max_iter=100, random_state=0).fit(X_train_smote, y_train_smote)

    print("La modèle est :", m)
    print("m.fit(X_train_smote, y_train_smote) \n") 
    
        #PREDICTIONS À PARTIR DU SMOTE
    test_pred_m=m.predict_proba(X_test)[:,1]
    train_pred_m=m.predict_proba(X_train_smote)[:,1]

        #SCORES DU 2E MODÈLE RESAMPLÉ
    train_score_m=m.score(X_train_smote, y_train_smote)
    test_score_m=m.score(X_test, y_test)
    
    print("train score :", train_score_m)
    print("test score :", test_score_m)

        #AUC
    print("\n")
    tpr_m, fpr_m, _=roc_curve(y_test, test_pred_m)
    roc_auc_m=auc(tpr_m, fpr_m)

    print("AUC : ",roc_auc_m)

        #CONFUSION MATRIX
    print("\n")
    test_pred_m_b=test_pred_m>0.5
    conf_mat_m=confusion_matrix(y_test, test_pred_m_b)
    print('Matrice de confusion :\n', conf_mat_m)
    
        #CLASSIFICATION REPORT
    print("\n")
    class_report_m= classification_report(y_test, test_pred_m_b)
    print("classification_report :\n",class_report_m)

    return m, m.coef_, roc_auc_m, class_report_m, conf_mat_m, train_score_m, test_score_m, train_pred_m, test_pred_m

m, m.coef_, roc_auc_m, class_report_m, conf_mat_m, train_score_m, test_score_m, train_pred_m, test_pred_m=modèle()


#Scoring à partir du modèle appliqué au test set_________________________________________________

#On attribue un certain nombre de points pour chaque classe : 

def points_par_classe(base, pdo):
    
    #CREATION DE LA GRILLE SCORE À PARTIR DES CLASSES, DU MODÈLE M ET CALIBRAGE SUR "base" POINTS
    points_par_classe = sc.scorecard(classes, m, xcolumns=X_train.columns, points0=base, pdo=pdo)

    return points_par_classe

points_par_classe=points_par_classe(base=1000, pdo=50)

points_par_classe.keys()

def points_par_classe_df():
    #TRANSFORMATION DES POINTS EN DATAFRAME PAR VARIABLES
    revol_bal=                         points_par_classe['revol_bal']
    home_ownership=                    points_par_classe['home_ownership']
    total_bal_ex_mort=                 points_par_classe['total_bal_ex_mort']
    verification_status=               points_par_classe['verification_status']
    mo_sin_old_il_acct=                points_par_classe['mo_sin_old_il_acct']
    installment=                       points_par_classe['installment']
    total_bc_limit=                    points_par_classe['total_bc_limit']
    term=                              points_par_classe['term']
    revol_util=                        points_par_classe['revol_util']
    mort_acc=                          points_par_classe['mort_acc']
    loan_amnt=                         points_par_classe['loan_amnt']
    fico=                              points_par_classe['fico']
    annual_inc=                        points_par_classe['annual_inc']
    inq_last_6mths=                    points_par_classe['inq_last_6mths']
    mo_sin_old_rev_tl_op =             points_par_classe['mo_sin_old_rev_tl_op']
    dti =                              points_par_classe['dti']
    
    return revol_bal, home_ownership, total_bal_ex_mort, verification_status, mo_sin_old_il_acct, installment,total_bc_limit,term, revol_util, mort_acc, loan_amnt, fico, annual_inc, inq_last_6mths, mo_sin_old_rev_tl_op, dti
    

revol_bal, home_ownership, total_bal_ex_mort, verification_status, mo_sin_old_il_acct, installment,total_bc_limit, term, revol_util, mort_acc, loan_amnt, fico, annual_inc, inq_last_6mths, mo_sin_old_rev_tl_op, dti = points_par_classe_df()         

#Puis les scores

def scores():

        #CALCUL DES SCORES TOTAUX DANS LE TRAIN SET
    train_score = sc.scorecard_ply(train_set, points_par_classe, print_step=0)
        #CALCUL DES SCORES DANS LE TEST SET
    test_score = sc.scorecard_ply(test_set, points_par_classe, print_step=0)
    
    return train_score, test_score

train_score, test_score=scores()

#SI ON VEUT VOIR LES SCORES PAR VARIABLES
train_score_vars= sc.scorecard_ply(train_set, points_par_classe, only_total_score=False, print_step=0)

score_avec_target = pd.concat([train_score, train_set['loan_status']],axis=1)
score_total       = score_avec_target['score']
bon_score         = score_avec_target[score_avec_target["loan_status"]== 0]['score']
mauvais_score     = score_avec_target[score_avec_target["loan_status"]== 1]['score']
    
def plot_score_distribution_train(score1,score2,label1,label2,title):
    sns.distplot(score1, color="darkturquoise", label=label1)
    sns.distplot(score2, color="tomato", label=label2)
    plt.legend(labels=[label1, label2])
    plt.title(title)
    plt.show()
    return 

plot_score_distribution_train(score1= bon_score ,score2=mauvais_score, label1='Bon',label2='Mauvais',title='Score Divergence Train')

score_avec_target = pd.concat([test_score, test_set['loan_status']],axis=1)
score_total= score_avec_target['score']
bon_score= score_avec_target[score_avec_target["loan_status"]==0]['score']
mauvais_score= score_avec_target[score_avec_target["loan_status"]==1]['score']
def plot_score_distribution_test(score1,score2,label1,label2,title):
    sns.distplot(score1, color="darkturquoise", label=label1)
    sns.distplot(score2, color="tomato", label=label2)
    plt.legend(labels=[label1, label2])
    plt.title(title)
    plt.show()
    return 

plot_score_distribution_test(bon_score,mauvais_score,label1='Bon',label2='Mauvais',title='Score Divergence Test')


# GRAPHIQUES DE PERFORMANCES

#POPULATION STABILITY INDEX (PSI)
sc.perf_psi(score = {'train':train_score, 'test':test_score}, 
        label = {'train':y_train, 'test':y_test},
        return_distr_dat=True)

#ROC & RECALL-PRECISION
train_perf_ROC = sc.perf_eva(y_train_smote, train_pred_m, title = "train", positive=1, plot_type=["roc", "pr"])
test_perf_ROC  = sc.perf_eva(y_test, test_pred_m, title = "test", positive=1, plot_type=["roc", "pr"])

#KOLMOGOROV - SMIRNOV & Lift
train_perf_KS = sc.perf_eva(y_train_smote, train_pred_m, title = "train", positive=1, plot_type=["ks", "lift"])
test_perf_KS = sc.perf_eva(y_test, test_pred_m, title = "test", positive=1, plot_type=["ks", "lift"])

# Comparaison avec des modèles de machine learning_____________________________________________

kf = KFold(n_splits=5)
print("DecisionTreeClassifier")
for fold in enumerate(kf.split(X), 1):
    model1 = DecisionTreeClassifier()
    model1.fit(X_train_smote, y_train_smote )  
    y_pred_dt = model1.predict_proba(X_test)[:,1]
    y_pred_dt_b= y_pred_dt >0.5
    print(f'Accuracy: {model1.score(X_test, y_test)}')
    print(f'f-score: {f1_score(y_test, y_pred_dt_b)}')
    print(f'recall-score: {recall_score(y_test, y_pred_dt_b)}')
    fpr_dt, tpr_dt, _=roc_curve(y_test, y_pred_dt)
    roc_auc_dt=auc(fpr_dt, tpr_dt)
    print("AUC", roc_auc_dt)
    print("\n")
    
print("\n\n")
print("RandomForestClassifier")
for fold in enumerate(kf.split(X), 1):
    model2 = RandomForestClassifier()
    model2.fit(X_train_smote, y_train_smote )  
    y_pred_rf = model2.predict_proba(X_test)[:,1]
    y_pred_rf_b= y_pred_rf >0.5
    print(f'Accuracy: {model2.score(X_test, y_test)}')
    print(f'f-score: {f1_score(y_test, y_pred_rf_b)}')
    print(f'recall-score: {recall_score(y_test, y_pred_rf_b)}')
    fpr_rf, tpr_rf, _=roc_curve(y_test, y_pred_rf)
    roc_auc_rf=auc(fpr_rf, tpr_rf)
    print("AUC", roc_auc_rf)
    print("\n")
    
    
print("\n\n")
print("GradientBoostingClassifier")
for fold in enumerate(kf.split(X), 1):
    model3 = GradientBoostingClassifier()
    model3.fit(X_train_smote, y_train_smote )  
    y_pred_gb = model3.predict_proba(X_test)[:,1]
    y_pred_gb_b= y_pred_gb >0.5
    print(f'Accuracy: {model3.score(X_test, y_test)}')
    print(f'f-score: {f1_score(y_test, y_pred_gb_b)}')
    print(f'recall-score: {recall_score(y_test, y_pred_gb_b)}')
    print("\n")
    fpr_gb, tpr_gb, _=roc_curve(y_test, y_pred_gb)
    roc_auc_gb=auc(fpr_gb, tpr_gb)
    print("AUC", roc_auc_gb) 

