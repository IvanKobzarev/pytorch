import scipy
from scipy import stats

import numpy as np
import sys

N1 = 20
MEAN1 = 188
STD1 = 10

N2 = 20
MEAN2 = 214
STD2 = 2

if (len(sys.argv) == 5):
    print("Data from sys.argv")
    MEAN1 = float(sys.argv[1])
    STD1 = float(sys.argv[2])
    MEAN2 = float(sys.argv[3])
    STD2 = float(sys.argv[4])

print("sample1: N1:{} MEAN1:{} STD1:{}".format(N1, MEAN1, STD1))
print("sample2: N2:{} MEAN2:{} STD2:{}".format(N2, MEAN2, STD2))

print("mean1-mean2:{}".format(MEAN1 - MEAN2))

t, p = scipy.stats.ttest_ind_from_stats(
    mean1=MEAN1, std1=STD1, nobs1=N1,
    mean2=MEAN2, std2=STD2, nobs2=N2,
    equal_var=False)

print("t:{:10.8f} p:{:10.8f}".format(t, p));


print("\n\n")

df = N1 + N2 - 2
N = (N1 + N2) / 2
for n in [N, 2 * N, 8* N]: 
    print("n:{}".format(n))
    df = 2 * n - 2
    for alpha in [0.01, 0.02, 0.05, 0.1]:
        tcv = scipy.stats.t.ppf(1.0 - alpha, df)
        stat_sig_diff = tcv * np.sqrt(STD1 * STD1 / n + STD2 * STD2 / n)
        t_alpha, p_alpha = scipy.stats.ttest_ind_from_stats(
            mean1=MEAN1, std1=STD1, nobs1=n,
            mean2=MEAN1 + stat_sig_diff, std2=STD2, nobs2=n,
            equal_var=False)
        print("alpha:{:.2f} stat_sig_diff:{:8.4f} p:{:8.4f}          tcv:{:4f} t:{:4f}"
                .format(alpha, stat_sig_diff, p_alpha, tcv, t_alpha))





