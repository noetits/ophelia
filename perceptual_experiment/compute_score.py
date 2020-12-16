import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# df.iloc[:,-8:]


def idx_to_case_xy(idx):
    x=((idx/100).astype(int)/100*5).astype(int)
    y=((idx-100*(idx/100).astype(int))/100*5).astype(int)

    return x,y


# df['Answer.selected_idx']
# df['Input.ref_idx']


X=np.load('X.npy')

w=(X[:,0].max()-X[:,0].min())
h=(X[:,1].max()-X[:,1].min())

def compute_scores(df):
    a_xs,a_ys=idx_to_case_xy(df['Answer.selected_idx'])
    i_xs,i_ys=idx_to_case_xy(df['Input.ref_idx'])
    dx=a_xs-i_xs
    dy=(a_ys-i_ys)*h/w
    scores=np.sqrt(dx**2+dy**2)
    return scores

def compute_score(df):
    return compute_scores(df).mean()



# worst baseline
ri_xs=np.random.randint(5, size=100000)
ri_ys=np.random.randint(5, size=100000)

# these should be equal with n>>>
x_max_delta=np.max([4-ri_xs, ri_xs], axis=0)
y_max_delta=np.max([4-ri_ys, ri_ys], axis=0)*h/w

max_score=np.sqrt(x_max_delta**2+y_max_delta**2).mean()
print('Worst baseline:', max_score)


# //!!\\ for these I should also do the *h/w if I want to use them

# other method for computing max_score: instead of random and max deltas,   just each case and max deltas
from itertools import product
n_cases=5
a=np.arange(n_cases)
cases=np.array(list(product(a, a*h/w)))
max_delta=np.max([[4,4*h/w]-cases, cases], axis=0)
# max_score=np.sqrt(x_max_delta**2+y_max_delta**2).mean()
max_score=np.sqrt((max_delta**2).sum(axis=1)).mean()

# baseline all in center
center_score=np.sqrt(((cases-np.array([2,2*h/w]))**2).sum(axis=1)).mean()
#  equal : look at the 25 possible distances in the square from the center
def dist(a,b):
    return np.sqrt(a**2+b**2)
y=h/w
(1+2+2*dist(1,y)+2*dist(2,y)+2*dist(2,2*y)+2*dist(1, 2*y)+y+2*y)*2/25

# random baseline
ri_xs=np.random.randint(5, size=100000)
ri_ys=np.random.randint(5, size=100000)
ra_xs=np.random.randint(5, size=100000)
ra_ys=np.random.randint(5, size=100000)
r_dx=ra_xs-ri_xs
r_dy=(ra_ys-ri_ys)*h/w
r_scores=np.sqrt(r_dx**2+r_dy**2)
r_score=r_scores.mean()

from scipy.stats import t

n=len(r_scores)
ts=t.ppf(0.975, n)

CI_r=r_scores.std()*ts/np.sqrt(n)

print('Random baseline:', r_score, ' +-', CI_r)
# print('center baseline:', center_score)

def print_scores_from_csv(df):
    score=compute_score(df)
    print('\n\n')
    print('Overall :', score)
    print('\n')
    for usr in pd.unique(df['Turkle.Username']):
        df_usr=df[df['Turkle.Username']==usr]
        print(usr, ': ', compute_score(df_usr))

# df=pd.read_csv('batch_25_same_sent-Batch_17_results_noe.csv')
# score=compute_score(df)
# print('noe:', score)

csv_path='batch_same_text_many-Batch_19_results.csv'
df=pd.read_csv(csv_path)
# print_scores_from_csv(df)

csv_path='batch_diff_text_many-Batch_20_results.csv'
df=pd.read_csv(csv_path)
# print_scores_from_csv(df)

csv_path='batch_isialab_same_text.csv'
df=pd.read_csv(csv_path)
# print_scores_from_csv(df)

csv_path='batch_victor_same.csv'
df=pd.read_csv(csv_path)
# print_scores_from_csv(df)

csv_path='batch_isialab_diff_text.csv'
df=pd.read_csv(csv_path)
# print_scores_from_csv(df)


paths_same_text=['batch_same_text_many-Batch_19_results.csv',
'batch_isialab_same_text.csv',
'batch_victor_same.csv']

paths_diff_text=['batch_diff_text_many-Batch_20_results.csv',
'batch_isialab_diff_text.csv']



dfs_same=[pd.read_csv(csv_path) for csv_path in paths_same_text]
dfs_diff=[pd.read_csv(csv_path) for csv_path in paths_diff_text]

df_same=pd.concat(dfs_same)
df_diff=pd.concat(dfs_diff)

# print_scores_from_csv(df_same)
# print_scores_from_csv(df_diff)

df_same['scores']=compute_scores(df_same)
df_diff['scores']=compute_scores(df_diff)

# df=pd.read_csv('batch_isialab_same_text.csv')
# score=compute_score(df_same)
# print(score)

# score=compute_score(df_diff)
# print(score)

fig, ax = plt.subplots()
scores_list=[df_same['scores'], df_diff['scores'], r_scores]
ax.boxplot(scores_list,showmeans=True,meanline=True)
methods=["same text", 'different text', 'Random']
ax.set_xticklabels(methods)
fig.savefig('boxplot.png')

plt.clf()
ax = sns.violinplot(df.iloc[:,0], y=df.iloc[:,1])
plt.savefig('violinplot.png')


# https://en.wikipedia.org/wiki/Confidence_interval
# https://en.wikipedia.org/wiki/Student%27s_t-distribution
#  We assume a normal distribution. We compute sample mean. 
# We do not know the true std and have to estimate it. 
# Therefore we use student's t distrib to have the "critical value" t* to obtain the CI:  t*s/sqrt(n)
#  the table is here: https://en.wikipedia.org/wiki/Student%27s_t-distribution#Table_of_selected_values

from scipy.stats import t
#  this is equal to the number written in the table at 97.5% (one-sided) or 95% (two-sided)
print(t.ppf(0.975, 60))

n=len(df_same['scores'])
ts=t.ppf(0.975, n)
CI_same=df_same['scores'].std()*ts/np.sqrt(n)

print('score same:', df_same['scores'].mean(), '+-', CI_same)

n=len(df_diff['scores'])
ts=t.ppf(0.975, n)
CI_diff=df_diff['scores'].std()*ts/np.sqrt(n)

print('score same:', df_diff['scores'].mean(), '+-', CI_diff)


def df_by_usr(df, info='scores'):
    d_usr_same={}
    for usr in pd.unique(df['Turkle.Username']):
        df_usr=df[df['Turkle.Username']==usr]
        # print(usr, compute_score(df_usr))
        d_usr_same[usr]=df_usr[info].tolist()
    df_usr=pd.DataFrame.from_dict(d_usr_same,orient='index').transpose()
    return df_usr


df_usr_same=df_by_usr(df_same, info='scores')
print('\n\n')
df_usr_diff=df_by_usr(df_diff, info='scores')


df_usr_same_duration=df_by_usr(df_same, info='WorkTimeInSeconds')
print('\n\n')
df_usr_diff_duration=df_by_usr(df_diff, info='WorkTimeInSeconds')



# https://en.wikipedia.org/wiki/Confidence_interval
# https://en.wikipedia.org/wiki/Student%27s_t-distribution
# We assume a normal distribution. We compute sample mean. 
# We do not know the true std and have to estimate it. 
# Therefore we use student's t distrib to have the "critical value" t* to obtain the CI:  t*s/sqrt(n)
# the table is here: https://en.wikipedia.org/wiki/Student%27s_t-distribution#Table_of_selected_values

from scipy.stats import t
# this is equal to the number written in the table at 97.5% (one-sided) or 95% (two-sided)
# print(t.ppf(0.975, 60))

def CI(df_usr):
    n=df_usr.count(axis=1)
    ts=t.ppf(0.975, n)
    CI=df_usr.std(axis=1)*ts/np.sqrt(n)
    return CI

CI(df_usr_diff)
#  to compute the mean \pm CI   by index,  n=number of participants that responded (non nan) = df_usr_diff.count(axis=1)
df_usr_diff.mean(axis=1)
n=df_usr_diff.count(axis=1)
ts=t.ppf(0.975, n)
CI_diff=df_usr_diff.std(axis=1)*ts/np.sqrt(n)

df_usr_same.mean(axis=1)
n=df_usr_same.count(axis=1)
ts=t.ppf(0.975, n)
CI_same=df_usr_same.std(axis=1)*ts/np.sqrt(n)

print('total answers same: ',df_usr_same.count().sum())
print('total answers diff: ',df_usr_diff.count().sum())


import seaborn as sns

def violin_by_idx(df, name='same', showmeans=True, showoutlier=True):
    x=[]
    y=[]
    df=df[:15]
    for column in df:
        # print(df_usr_diff[column])
        x+=df[column].index.tolist()
        y+=df[column].tolist()
    data=pd.DataFrame([x,y]).T.dropna()

    scores_list=[]
    for i in range(len(df)):
        scores_list.append(data[data.iloc[:,0]==i].iloc[:,1].tolist())
    
    plt.clf()
    if showoutlier:
        result=plt.boxplot(scores_list,showmeans=showmeans,meanline=True)
    else:
        result=plt.boxplot(scores_list,showmeans=showmeans,meanline=True,sym='')
    # ax.set_xticklabels(range(df.shape[-1])
    plt.savefig('boxplot_by_idx_'+name+'.png')

    plt.clf()
    ax = sns.violinplot(x=data.iloc[:,0], y=data.iloc[:,1] , cut=0)
    plt.savefig('violinplot_'+name+'.png')

    return result


def stat_list(result, stat='means'):
    return [el._y[0] for el in result[stat]]

result_same=violin_by_idx(df_usr_same)
result_diff=violin_by_idx(df_usr_diff, name='diff')

result_same_duration=violin_by_idx(df_usr_same_duration, name='same_duration', showmeans=False, showoutlier=False)
result_diff_duration=violin_by_idx(df_usr_diff_duration, name='diff_duration', showmeans=False, showoutlier=False)

means_same=stat_list(result_same, stat='means')
means_diff=stat_list(result_diff, stat='means')
medians_same_duration=stat_list(result_same_duration, stat='medians')
medians_diff_duration=stat_list(result_diff_duration, stat='medians')

boxes_diff_duration=[el._y for el in result_diff_duration['boxes']]

from scipy.stats import linregress



print('means same:',linregress(range(len(means_same)),means_same))
print('means diff:',linregress(range(len(means_diff)),means_diff))

print('medians duration same:',linregress(range(len(medians_same_duration)),medians_same_duration))
print('medians duration diff:',linregress(range(len(medians_diff_duration)),medians_diff_duration))

# plt.clf()
# ax = sns.lineplot(x=x, y=y)
# plt.savefig('lineplot.png')


# plt.clf()
# ax = sns.regplot(x=x, y=y)
# plt.savefig('regplot.png')

# df=pd.DataFrame([x,y]).T.dropna()
# plt.clf()
# ax = plt.hist2d(x=df.iloc[:,0], y=df.iloc[:,1],  bins=(10, 5))
# plt.savefig('hist2d.png')


# t-test to see if means are different
# https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f