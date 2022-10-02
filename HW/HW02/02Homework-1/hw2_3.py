#%%

# 有空的話，將 np.polyfit改成 np.polynomial

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import statsmodels.api as sm
import scipy.stats as stats

#%%
# read data

# row data: 19900531  7.571  7.931  7.977  7.996  8.126  8.247  8.298  8.304  8.311  8.327  8.369  8.462  8.487  8.492  8.479  8.510  8.507  8.404

data_x_1990 = [1, 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120]
data_y_1990 = [7.571,  7.931,  7.977,  7.996,  8.126,  8.247,  8.298,  8.304,  8.311,  8.327,  8.369,  8.462,  8.487,  8.492,  8.479,  8.510,  8.507,  8.404]


#%%
#================= part a =================
plt.scatter(data_x_1990, data_y_1990, c="red")
plt.xlabel("Maturity")
plt.ylabel("Yields")
plt.title("Yields vs maturity (data: 19900531)")
plt.show()

#%%
#================= part b =================

# plot fit result
r2_list = []
plt.figure(figsize=(15,12))
for order in range(1, 7):

    plt.subplot(2,3,order)
    fit = np.polyfit(data_x_1990, data_y_1990, deg=order) # 用n次多項式擬合
    fit_model = np.poly1d(fit)
    fit_y = fit_model(data_x_1990)

    plt.plot(data_x_1990, data_y_1990, '*', label='original values')
    plt.plot(data_x_1990, fit_y, 'r', label='polyfit values')
    plt.xlabel('maturity')
    plt.ylabel('y axis')
    plt.legend(loc=4) # 指定圖例象限的位置
    plt.title(f'order: {order}')
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    r2_list.append(r2_score(y_true=data_y_1990, y_pred=fit_model(data_x_1990))) # 計算r2數值

plt.show()

#%%
# plot R2 score
x = [i for i in range(1, 7)]

plt.plot(x, r2_list, 'r*-')
plt.xlabel('Order k')
plt.ylabel('R2 value')
plt.title('R2 vs  the polynomial order k')
plt.show()

# %%
#================= part c =================

# coefficients, residuals, rank, singular_values, rcond = np.polyfit(data_x_1990, data_y_1990, deg=4, full=True) # 用n次多項式擬合
coefficients = np.polyfit(data_x_1990, data_y_1990, deg=4) # 用n次多項式擬合
fit_model_4 = np.poly1d(coefficients)
fit_y = fit_model_4(data_x_1990)

# 計算residuals
residuals = []
for i, y_test in enumerate(data_y_1990):
    residuals.append(y_test - fit_y[i])

plt.scatter(data_x_1990, residuals, c='r')
plt.xlabel('Maturity')
plt.ylabel('Residual ')
plt.title('Residual plot')
plt.show()

# %%
#================= part d =================

## histogram
plt.hist(residuals, bins='auto')
plt.title("Histogram of the residuals")
plt.show()

## QQ plot
residuals = np.array(residuals)
fig = sm.qqplot(residuals, stats.t, fit=True, line="45")
plt.title("quantile-quantile plot of the residuals")
plt.show()