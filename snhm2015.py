# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:05:06 2018

@author: TilkeyYang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



# Enter directory to change dir
os.chdir('.') # Enter your directory
#os.chdir('D:/Github/Visualisation/snhm2015') #This one is my dir
cwd = os.getcwd()
print('Working Directory:', cwd)
os.makedirs(cwd + '/figures', exist_ok=True)


# Autolabel function
def autolabel(rects, nb_rects, rot=0):
    i = 0  
    for rect in rects:
      if i < nb_rects:
        # Calculate automatically the height
        height = rect.get_height() + 0.5
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, 
                 '%s' % float(height), size=8, family='serif', style='italic', 
                 rotation=rot)
        i+=1
        
# Auto set frame spines
def set_spines():
  framex = plt.gca()
  framex.spines['top'].set_visible(False)
  framex.spines['right'].set_visible(False)
  framex.spines['left'].set_visible(True)
  framex.spines['left'].set_linewidth(1.2)
  framex.spines['left'].set_linestyle('--')
  framex.spines['bottom'].set_visible(True)
  framex.spines['bottom'].set_linewidth(1.2)
  framex.spines['bottom'].set_linestyle('--')


        
# =============================================================================
# Initialization
# =============================================================================
# Import csv
data = pd.read_csv('data/sal_sex_rev_15.csv',sep=';', skiprows = 1,
                   names=[
                   'CODGEO', 'LIBGEO', 'SNHM15', 'C15', 'P15', 'E15', 'O15', 
                   'SNHMF15', 'FC15', 'FP15', 'FE15', 'FO15', 'SNHMH15', 
                   'HC15', 'HP15', 'HE15', 'HO15', 'Age1815', 'Age2615', 
                   'Age5015', 'AgeF1815', 'AgeF2615', 'AgeF5015', 'AgeH1815',
                   'AgeH2615', 'AgeH5015', 'H-F15C', 'H-F15P', 'H-F15E', 'H-F15O'
                   ])

data_ecart = pd.read_csv('data/ecart_avg_15.csv',sep=';', skiprows = 1,
                 names=['snhm', 'ecart', 'Emploi'])

# Change plotting style
plt.style.use('dark_background')



# =============================================================================
# FIGURE1
# Compare the M/F Salary in each department of France
# =============================================================================
fig_hf15 = plt.figure()

# Export useful columns: Region names, Region's Male Salary, Region's Femail Salary
libgeo = data.LIBGEO
f15 = data.SNHMF15
h15 = data.SNHMH15

# Define width for the bar plot
total_width, n = 0.8, 2
width = total_width / n

# Create X abscisse
x =list(range(len(f15)))

# Create bar plots
barF = plt.bar(x, f15, width=width, label='FEMME', color = 'C3', 
               edgecolor = None, linewidth = 0)
for i in range(len(x)):
    x[i] = x[i] + width
barH = plt.bar(x, h15, width=width, label='HOMME',fc = 'c', 
               edgecolor = None, linewidth = 0)

# Affiche labels
autolabel(barF, 17, 90)
autolabel(barH, 17, 90)

# Frame format
set_spines()
frame1 = plt.gca()
frame1.spines['left'].set_visible(False)

# Graph format
plt.xticks(np.arange(20), data.LIBGEO, rotation=90, fontsize=7.5)
plt.locator_params(nbins=20)
plt.axis([-1, 17, 10, 20])
plt.ylim(10, 20)
plt.xlim(-0.2, 16.5)
plt.grid(False, linestyle = "-", color = "w", linewidth = "0")
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', 
                labelleft='off', labeltop='off', labelright='off', labelbottom='on')

# Save figure
plt.savefig(cwd + '/figures/Region_Sex.png', 
            format='png', bbox_inches='tight', transparent=True, dpi=600) 
plt.show()



# =============================================================================
# FIGURE2
# Compare the salary of 'Cadres', 'Professionals', 'Employés', 'Ouvriers'
# =============================================================================
fig_cpeo15 = plt.figure()

# Extract and trans CPEO_H/F from global daraframe
cpeo = pd.DataFrame(data, columns=['FC15', 'HC15', 'FP15', 'HP15',
                                   'FE15', 'HE15', 'FO15', 'HO15'])
cpeo_t = cpeo.T
cpeo_t.columns = libgeo
cpeo_rouen = pd.DataFrame(cpeo_t, columns=['France metropolitaine','Normandie',
                                           'Seine-Maritime','Rouen']).T
cpeoFR = pd.DataFrame(cpeo_rouen , columns=['FC15', 'FP15', 'FE15', 'FO15']).T
cpeoHR = pd.DataFrame(cpeo_rouen , columns=['HC15', 'HP15', 'HO15', 'HE15']).T

# Raname for columns
columns = ['France metropolitaine','Normandie','Seine-Maritime','Rouen']

# Initialize the vertical-offset for the stacked bar chart
y_offset = np.zeros(len(columns))*2
femmeX = np.arange(len(columns))*2-1 + 1.2
hommeX = np.arange(len(columns))*2 + 1.2

# Get some pastel shades for the color
region_emplois = ['France \nmetropolitaine (F)','France \nmetropolitaine (H)',
                  'Normandie (H)','Normandie (H)','Seine-Maritime (F)',
                  'Seine-Maritime (H)','Rouen (F)','Rouen (H)']

# Define width for the bar plot
bar_width = 0.65

# Define color for the bar plot
colorsF = plt.cm.PuRd(np.linspace(0.75, 0.25, 4))
colorsH = plt.cm.PuBuGn(np.linspace(0.75, 0.25, 4))

# Create bar plots
for i in range(0, len(cpeoFR)):
    barEmploiF = plt.bar(femmeX, cpeoFR.iloc[i], bar_width, bottom=y_offset, 
                         color=colorsF[i], edgecolor = None, linewidth = 0)
    if i < 3: autolabel(barEmploiF, 4)
for i in range(0, len(cpeoHR)):
    barEmploiH = plt.bar(hommeX, cpeoHR.iloc[i], bar_width, bottom=y_offset, 
                         color=colorsH[i], edgecolor = None, linewidth = 0)
    if i < 3: autolabel(barEmploiH, 4)

# Frame format
set_spines()
frame2 = plt.gca()
frame2.spines['left'].set_visible(False)
    
# Graph format
plt.xticks(np.arange(8), region_emplois, rotation=50, fontsize=9)
plt.ylim(0, 30)
plt.grid(True, linestyle = "-", color = "w", linewidth = "0.1")
plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', 
                labelleft='off', labeltop='off', labelright='off', labelbottom='on')

# Save figure
plt.savefig('./figures/Emploi_Sex.png', 
            format='png', bbox_inches='tight', transparent=True, dpi=600)
plt.show()



# =============================================================================
# FIGURE3
# Compare the salary of different age groups
# =============================================================================
fig_age15 = plt.figure()

# Extract and trans age groups' data from global daraframe
age = pd.DataFrame(data, columns=['Age1815', 'Age2615', 'Age5015', 'AgeF1815', 
                                  'AgeF2615', 'AgeF5015', 'AgeH1815',
                                   'AgeH2615', 'AgeH5015'])
age_t = age.T
age_t.columns = libgeo
age_rouen = pd.DataFrame(age_t, columns=['France metropolitaine','Normandie',
                                         'Seine-Maritime','Rouen'])
age_rouen.columns=['France','Normandie','Seine-Maritime','Rouen']
age_rouen_T = age_rouen.T
ageH = pd.DataFrame(age_rouen_T, columns=['AgeH1815', 'AgeH2615', 'AgeH5015']).T
ageF = pd.DataFrame(age_rouen_T, columns=['AgeF1815', 'AgeF2615', 'AgeF5015']).T
ageTout = pd.DataFrame(age_rouen_T, columns=['Age1815', 'Age2615', 'Age5015']).T
ageH.columns=['FranceMetropol_H','Normandie_H','SeineMaritime_H','Rouen_H']
ageF.columns=['FranceMetropol_F','Normandie_F','SeineMaritime_F','Rouen_F']

# Create X abscisse
ageX = np.arange(3)

# Create plots
plt.plot(ageX, ageH.FranceMetropol_H, color = '#8EE5EE', marker = '*')
plt.plot(ageX, ageH.Normandie_H, color = '#53868B', marker = '*')
plt.plot(ageX, ageH.SeineMaritime_H, color = '#6E8B3D', marker = '*')
plt.plot(ageX, ageH.Rouen_H, color = '#CAFF70', marker = '*')
plt.plot(ageX, ageF.FranceMetropol_F, color = '#FF4500', marker = '*')
plt.plot(ageX, ageF.Normandie_F, color = '#CD8162', marker = '*')
plt.plot(ageX, ageF.SeineMaritime_F, color = '#7D26CD', marker = '*')
plt.plot(ageX, ageF.Rouen_F, color = '#FFA07A', marker = '*')

# Frame format
set_spines()
    
# Graph format
plt.grid(True, linestyle = "-", color = "w", linewidth = "0.08")
plt.legend(loc=2,prop={'size':9})

# Create xticks
age_sex = ['18 à 26 ans','26 à 50 ans','plus de 50 ans']
plt.xticks(np.arange(3), age_sex, rotation=0, fontsize=9)

# Save figure
plt.savefig('./figures/Age_Sex.png', 
            format='png', bbox_inches='tight', transparent=True, dpi=600)
plt.show()



# =============================================================================
# FIGURE4
# Linear Regression of 'difference of salary' and 'average salary'
# =============================================================================

# Plotting regression
reg_pts = sns.regplot(x="snhm", y="ecart", data=data_ecart, 
                      scatter_kws={'alpha':0.01})
plt.grid(False)
# Adding scatterplot
fig_pts = sns.scatterplot(x="snhm", y="ecart", hue="Emploi", palette="Set2", 
                          data=data_ecart, alpha=0.5)
plt.grid(False)
plt.legend(loc=2,prop={'size':10})

# Frame format
set_spines()
frame4 = plt.gca()
frame4.set_ylabel('')    
frame4.set_xlabel('')

# Save figure
plt.tight_layout()
plt.savefig('./figures/Ecart_Salaire.png', 
            format='png', bbox_inches='tight', transparent=True, dpi=600)
plt.show()



# =============================================================================
# FIGURE5
# Circles
# =============================================================================

# Create a 67.7% circle for female
circF = plt.figure()  
femmeAC = [67.7, 32.3]  
# Define color
colorsF = ['red','rosybrown'] 
plt.pie(femmeAC, colors=colorsF, shadow = False, startangle = -90) 
# Make sure plot a circle by limitting x = y 
plt.axis('equal') 
# Save figure
plt.savefig('./figures/femmeAC.png', 
            format='png', bbox_inches='tight', transparent=True, dpi=600)
plt.show()


# Create a 75.5% circle for male
circH = plt.figure()
hommeAC = [75.5, 24.5]  
colorsH = ['darkturquoise','slategray']
plt.pie(hommeAC, colors=colorsH, shadow = False, startangle = -90) 
plt.axis('equal') 
# Save figure
plt.savefig('./figures/hommeAC.png', 
            format='png', bbox_inches='tight', transparent=True, dpi=600)
plt.show()
