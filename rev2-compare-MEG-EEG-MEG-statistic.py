# %%
import numpy as np
import pandas as pd

from scipy import stats
from itertools import combinations
from IPython.display import display
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

# %%
table = pd.read_csv('./data/MVPA.FBCSP.compare.MEG-EEG.vote.results/df_compare.csv', index_col=0)
table

# %%
# 1. Select MEG records
df_meg = table[
    (table['modality'] == 'MEG') &
    (table['chMark'].isin(['C', 'F', 'O', 'P', 'T'])) &
    (table['chHemi'].isin(['L', 'R', 'B']))
].copy()

# Two-way ANOVA
model = ols(
    'acc ~ C(chMark) * C(chHemi)',
    data=df_meg
).fit()

anova_results = anova_lm(model, typ=2)

print('Two-way ANOVA (MEG)')
display(anova_results.round(5))

anova_results.to_csv('./data/MVPA.FBCSP.compare.MEG-EEG.vote.results/anova_meg.csv')

# %%
# 2. Pairwise paired t-tests for chHemi = B

from itertools import combinations
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Select bilateral hemisphere
df_b = df_meg[df_meg['chHemi'] == 'B'].copy()

# Pivot: rows = subjects, columns = chMark
wide = df_b.pivot(
    index='subject',
    columns='chMark',
    values='acc'
)

marks = ['C', 'F', 'O', 'P', 'T']
wide = wide[marks]

# Pairwise paired t-tests
results = []

for mark1, mark2 in combinations(marks, 2):

    paired = wide[[mark1, mark2]].dropna()

    t_stat, p_value = stats.ttest_rel(
        paired[mark1],
        paired[mark2]
    )

    diff = paired[mark1] - paired[mark2]

    results.append({
        'Comparison': f'{mark1} vs {mark2}',
        'N': len(paired),
        'Mean_1': paired[mark1].mean(),
        'Mean_2': paired[mark2].mean(),
        'Mean_Diff': diff.mean(),
        't': t_stat,
        'df': len(paired) - 1,
        'p': p_value
    })

# Results DataFrame
ttest_results = pd.DataFrame(results)

# Multiple comparison correction
ttest_results['p_holm'] = multipletests(
    ttest_results['p'],
    method='holm'
)[1]

ttest_results['p_fdr'] = multipletests(
    ttest_results['p'],
    method='fdr_bh'
)[1]

# Significance
ttest_results['Significant'] = (
    ttest_results['p_holm'] < 0.05
)

print('Pairwise paired t-tests (MEG, chHemi=B)')
display(ttest_results.round(5))
ttest_results.to_csv('./data/MVPA.FBCSP.compare.MEG-EEG.vote.results/ttest_meg.csv')

# %%
