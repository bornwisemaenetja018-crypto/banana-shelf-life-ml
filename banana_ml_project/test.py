import matplotlib.pyplot as plt
import scipy.stats as stats

ethylene = [2.1, 2.5, 3.0, 3.8, 4.2, 5.1, 5.8,
6.0, 6.5, 7.0, 7.2, 7.8, 8.1, 8.5]

stats.probplot(ethylene, dist="norm", plot=plt)

plt.title('Q-Q Plot of Ethylene Concentration')
plt.grid(True)
plt.show()

import scipy.stats as stats

stat, p = stats.shapiro(ethylene)

print("Shapiro-Wilk Statistic =", stat)
print("p-value =", p)

if p > 0.05:
    print("Data are normally distributed")
else:
    print("Data are not normally distributed")