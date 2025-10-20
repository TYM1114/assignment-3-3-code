以下是assignment 1-1的1-1.py code
```bash
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t

temp = [0, 0, 0, 15, 15, 15, 30, 30, 30, 45, 45, 45, 60, 60, 60, 75, 75, 75] #temperature
dweight = [7, 8, 9, 11, 14, 13, 24, 25, 22, 33, 29, 30, 45, 40, 43, 49, 47, 50] #dissolve weight

temp = np.array(temp)
weight = np.array(dweight)

# Calculate correlation coefficient
correlation = np.corrcoef(temp, weight)[0, 1] #return a matrix [1.0 , coefficient]
                                              #                [coefficient , 1.0]
print("CORRELATION ANALYSIS")
print(f"Correlation coefficient (r): {correlation:.4f}")
print()

# Linear regression
#slope = beta1 ,intercept = beta0 ,r_value = correlation coefficient
slope, intercept, r_value, p_value, std_err = stats.linregress(temp, weight) # use this instead polyfit ref :https://stackoverflow.com/questions/31126698/differences-between-scipy-stats-linregress-numpy-polynomial-polynomial-polyfi

print("-"*20)
print("LINEAR REGRESSION")
print(f"Regression equation: y = {intercept:.4f} + {slope:.4f}x")
print(f"R-squared: {r_value**2:.4f}")
print()

# Calculate std_err and intervals
n = len(temp)  #sample = 18
residuals = weight - (intercept + slope * temp) 
mse = np.sum(residuals**2) / (n - 2) # SSE / DF
se = np.sqrt(mse)#_std_err

x_mean = np.mean(temp)
sxx = np.sum((temp - x_mean)**2)

# Prediction at temp = 40
x_pred = 40
y_pred = intercept + slope * x_pred  # y = beta0 + beta1 *x


# equation : CI = y_pred ± t × se × sqrt[1/n + (40 - x_mean)²/Sxx]
# Standard error for mean prediction (confidence interval)
se_mean = se * np.sqrt(1/n + (x_pred - x_mean)**2 / sxx)

# equation : PI =  y_pred ± t × se × sqrt[1 + 1/n + (40 - x_mean)²/Sxx]
# Standard error for individual prediction (prediction interval)
se_pred = se * np.sqrt(1 + 1/n + (x_pred - x_mean)**2 / sxx)

# t value for 95% cl
alpha = 0.05
df = n - 2 # because we estimated intercept & slope 
t_crit = t.ppf(1 - alpha/2, df)

# Calculate intervals
ci_low = y_pred - t_crit * se_mean
ci_up = y_pred + t_crit * se_mean

pi_low = y_pred - t_crit * se_pred
pi_up = y_pred + t_crit * se_pred

print("-"*20)
print("PREDICTION AT temp = 40°C")
print(f"Point prediction: {y_pred:.4f} grams")
print()
print(f"95% Confidence Interval:")
print(f"  [{ci_low:.4f}, {ci_up:.4f}]")
print()
print(f"95% Prediction Interval:")
print(f"  [{pi_low:.4f}, {pi_up:.4f}]")
print()

# Create scatter plot with regression line
plt.figure(figsize=(8,6))

# scatter 
plt.scatter(temp, weight, alpha=1.0, s=50, label='Observed data')

# regression
x_line = np.linspace(0, 80, 100)
y_line = intercept + slope * x_line
plt.plot(x_line, y_line, 'r-', linewidth=2, 
         label=f'Regression line: y = {intercept:.2f} + {slope:.2f}x')

plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Dissolved Weight (grams)', fontsize=12)
plt.title('Temperature vs Dissolved Weight with Regression Line', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

```
