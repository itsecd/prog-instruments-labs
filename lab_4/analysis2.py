import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='lab_4/program_log.log',
    filemode='w'
)

logging.info("Program started.")

import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sb

# Заданные параметры для 9 варианта
a = - 4
sigma2 = 6
std_sigma = np.sqrt(sigma2)
gamma = 0.90
n = 17
M = 1800

logging.info(f"Parameters initialized: a={a}, sigma2={sigma2}, gamma={gamma}, n={n}, M={M}")

sample = np.random.normal(loc = a, scale = std_sigma, size = n)
logging.info(f"Sample generated: {sample}")

t_gamma = stats.norm.ppf((1 + gamma) / 2)
mean_sample = np.mean(sample)

error_known = t_gamma * (std_sigma / np.sqrt(n))
interval_mean_known = (
    mean_sample - error_known,
    mean_sample + error_known
)

logging.info(f"Confidence interval for mean (known variance): {interval_mean_known}")

interval_mean_known_py = stats.norm.interval(gamma, loc=mean_sample, scale=std_sigma/np.sqrt(n))
logging.info(f"Check confidence interval (known variance) with built-in function: {interval_mean_known_py}")

t_critical = stats.t.ppf((1 + gamma) / 2, df = n - 1)
std_sample = np.std(sample, ddof = 1)

error_unknow = t_critical * (std_sample / np.sqrt(n))
interval_mean_unknown = (
    mean_sample - error_unknow,
    mean_sample + error_unknow
)

logging.info(f"Confidence interval for mean (unknown variance): {interval_mean_unknown}")

interval_mean_unknown_py = stats.t.interval(gamma, df=n-1, loc=mean_sample, scale=std_sample/np.sqrt(n))
logging.info(f"Check confidence interval (unknown variance) with built-in function: {interval_mean_unknown_py}")

chi2_lower = stats.chi2.ppf((1 - gamma) / 2, df = n - 1)
chi2_upper = stats.chi2.ppf((1 + gamma) / 2, df = n - 1)

var_lower = (n - 1) * (std_sample**2) / chi2_upper
var_upper = (n - 1) * (std_sample**2) / chi2_lower
interval_var = (var_lower, var_upper)

logging.info(f"Confidence interval for variance: {interval_var}")

chi2_interval = stats.chi2.interval(gamma, df=n-1)
var_lower_py = (n - 1) * (std_sample ** 2) / chi2_interval[1]
var_upper_py = (n - 1) * (std_sample ** 2) / chi2_interval[0]
interval_var_py = (var_lower_py, var_upper_py)
logging.info(f"Check confidence interval for variance with built-in function: {interval_var_py}")

confidence_levels = np.linspace(0.80, 0.99, 50)
interval_lengths_mean_known_variance = []
interval_lengths_variance_unknown_mean = []
interval_lengths_variance = []

for gamma_ in confidence_levels:
    # Для математического ожидания при известной дисперсии, но неизвестном математическом ожидании
    t_gamma_ = stats.norm.ppf((1 + gamma_) / 2)
    error_known_mean_ = t_gamma_ * (std_sigma / np.sqrt(n))
    interval_length_mean_known_variance = 2 * error_known_mean_
    interval_lengths_mean_known_variance.append(interval_length_mean_known_variance)

    # Для дисперсии при неизвестном мат. ожидании
    chi2_lower_ = stats.chi2.ppf((1 - gamma_) / 2, df=n-1)
    chi2_upper_ = stats.chi2.ppf((1 + gamma_) / 2, df=n-1)
    interval_length_variance_unknown_mean = (n - 1) * (std_sample ** 2) * (1/chi2_lower_ - 1/chi2_upper_)
    interval_lengths_variance_unknown_mean.append(interval_length_variance_unknown_mean)

    # Для сигмы^2
    interval_length_variance = chi2_upper_ - chi2_lower_
    interval_lengths_variance.append(interval_length_variance)

logging.info("Plotting interval lengths for confidence levels...")
plt.figure(figsize=(12, 6))
plt.plot(confidence_levels, interval_lengths_mean_known_variance, label='Interval length for mean (known variance)')
plt.plot(confidence_levels, interval_lengths_variance_unknown_mean, label='Interval length for variance (unknown mean)')
plt.plot(confidence_levels, interval_lengths_variance, label='Interval length for sigma^2')
plt.xlabel('Confidence level')
plt.ylabel('Interval length')
plt.title('Dependence of interval length on confidence level')
plt.legend()
plt.grid()
plt.show()

sample_sizes_3 = np.arange(10, 101, 5)
interval_lengths_mean_known_variance_3 = []
interval_lengths_variance_unknown_mean_3 = []
interval_lengths_variance_3 = []

for n_ in sample_sizes_3:
    error_known_mean_ = t_gamma * (std_sigma / np.sqrt(n_))
    interval_length_mean_known_variance = 2 * error_known_mean_
    interval_lengths_mean_known_variance_3.append(interval_length_mean_known_variance)

    chi2_lower_ = stats.chi2.ppf((1 - gamma) / 2, df=n_-1)
    chi2_upper_ = stats.chi2.ppf((1 + gamma) / 2, df=n_-1)
    interval_length_variance_unknown_mean = (n_ - 1) * (std_sample ** 2) * (1/chi2_lower_ - 1/chi2_upper_)
    interval_lengths_variance_unknown_mean_3.append(interval_length_variance_unknown_mean)

    interval_length_variance = 2 * (n_ - 1) * (std_sample ** 2) * (1/chi2_lower_ - 1/chi2_upper_)
    interval_lengths_variance_3.append(interval_length_variance)

logging.info("Plotting interval lengths for sample sizes...")
plt.figure(figsize=(12, 6))
plt.plot(sample_sizes_3, interval_lengths_mean_known_variance_3, label='Interval length for mean (known variance)')
plt.plot(sample_sizes_3, interval_lengths_variance_unknown_mean_3, label='Interval length for variance (unknown mean)')
plt.plot(sample_sizes_3, interval_lengths_variance_3, label='Interval length for sigma^2')
plt.xlabel('Sample size')
plt.ylabel('Interval length')
plt.title('Dependence of interval length on sample size')
plt.legend()
plt.grid()
plt.show()

logging.info("Generating histogram of combined samples...")
all_samples = np.concatenate([np.random.normal(loc=a, scale=std_sigma, size=n) for _ in range(M)])
plt.hist(all_samples, bins=20, density=True, alpha=0.9, color='g', label='Sample histogram')

x = np.linspace(min(all_samples), max(all_samples), 1000)
pdf = stats.norm.pdf(x, loc=a, scale=std_sigma)

plt.plot(x, pdf, 'r-', lw=2, label='Normal distribution density')
plt.xlabel('Value')
plt.ylabel('Probability density')
plt.title('Histogram and normal distribution density')
plt.legend()
plt.grid()
plt.show()

M = 1800
Z_values = []

for _ in range(M):
    sample_m = np.random.normal(loc = a, scale = std_sigma, size = n)
    s2_sample = np.var(sample_m, ddof = 1)
    Z = (n - 1) * s2_sample / sigma2 
    Z_values.append(Z)

mean_Z = np.mean(Z_values)
variance_Z = np.var(Z_values, ddof = 1)
median_Z = np.median(Z_values)
curt_Z = stats.kurtosis(Z_values)
std_Z = np.std(Z_values)

logging.info(f"Mean Z: {mean_Z}")
logging.info(f"Variance Z: {variance_Z}")
logging.info(f"Median Z: {median_Z}")
logging.info(f"Kurtosis Z: {curt_Z}")
logging.info(f"Standard deviation Z: {std_Z}")

plt.figure(figsize=(12, 6))
count, bins, _ = plt.hist(Z_values, bins=20, density=True, alpha=0.6, color='b', label='Z histogram')

x = np.linspace(min(Z_values), max(Z_values), 1000)
pdf_chi2 = stats.chi2.pdf(x, df = n - 1)
plt.plot(x, pdf_chi2, 'r-', lw = 2, label = 'Theoretical chi-squared density')

plt.xlabel('Value Z')
plt.ylabel('Probability density')
plt.title('Histogram of relative frequencies and theoretical chi-squared density')
plt.legend()
plt.grid()
plt.show()

logging.info("Generating boxplot for Z values...")
sb.boxplot(x = Z_values)
plt.title('Boxplot for chi-squared distribution')
plt.grid()
plt.show()

logging.info("Program completed.")
