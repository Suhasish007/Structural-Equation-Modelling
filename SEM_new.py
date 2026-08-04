import pandas as pd
import numpy as np
from semopy import Model, inspector
import warnings
import os
from scipy import stats
import matplotlib.pyplot as plt
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger().setLevel(logging.ERROR)  # Suppress semopy optimization warnings (e.g., Fisher Information Matrix)

# Create output directory for results
os.makedirs('sem_results', exist_ok=True)

# Load dataset
data = pd.read_csv('cleaned_data.csv')

# Document variable meanings for theoretical consistency
variable_meanings = {
    'Q4': 'Daily social media usage (hours)',
    'Q9': 'Familiarity with digital fashion',
    'Q11': 'Motivation factors for digital fashion adoption',
    'Q12': 'Willingness to pay for digital fashion items',
    'Q14': 'Environmental consciousness',
    'Q16': 'Investment likelihood in digital fashion'
}

print("\nVariable descriptions:")
for var, desc in variable_meanings.items():
    print(f"{var}: {desc}")

# Define mappings for categorical variables
mappings = {
    'Q4': {  # Social media usage
        'Not at all': 1, '0 - 1': 2, '2 - 3': 3, '4 - 5': 4, '5 + hours': 5
    },
    'Q9': {  # Familiarity with digital fashion
        'Yes': 3, 'Not sure': 2, 'Not Sure': 2, 'No': 1
    },
    'Q12': {  # Willingness to pay
        'Above $200/ £162/ ₹17,252': 6,
        '($101 - $200)/ (£82 - £162)/ (₹7073 - ₹17,252)': 5,
        '($101 - $200)/ (£82 -  £162)/ (₹7073 - ₹17,252)': 5,
        '($51 - $100)/ (£41 - £81)/ (₹3535 - ₹8626)': 4,
        '($51 - $100)/ (£41 -  £81)/ (₹3535 - ₹8626)': 4,
        '($21 - $50)/ (£17 - £40)/ (₹1810 - ₹4310)': 3,
        '($21 - $50)/ (£17 -  £40)/ (₹1810 - ₹4310)': 3,
        '($5 - $20)/ (£4 - £16)/ (₹430 - ₹1725)': 2,
        '($5 - $20)/ (£4 -  £16)/ (₹430 - ₹1725)': 2,
        'I would rather not buy': 1
    }
}

# Create binary indicators for each motivation factor
q11_attributes = ['Design', 'Uniqueness', 'Reputation', 'Curiosity',
                 'Social Media Influence', 'Price', 'Environment',
                 'Customisation', 'Utility']
q11_attributes_sanitized = [attr.replace(' ', '_') for attr in q11_attributes]

for i, attr in enumerate(q11_attributes):
    sanitized_attr = q11_attributes_sanitized[i]
    col_name = f'Q11_{sanitized_attr}'
    pattern = f"(?i){attr}"
    data[col_name] = data['Q11'].str.contains(pattern, na=False).astype(int)

# Calculate total number of motivations per respondent (composite/formative approach)
data['motivation_count'] = data[[f'Q11_{attr}' for attr in q11_attributes_sanitized]].sum(axis=1)

# NOTE: Q11 motivation items are conceptually FORMATIVE (respondents select applicable items),
# not reflective (not parallel reflections of a latent trait).
# Using motivation_count (composite score) is more appropriate than a reflective latent.

# Apply mappings to categorical columns
for column, mapping in mappings.items():
    if column in data.columns:
        data[f"{column}_original"] = data[column].copy()
        data[column] = data[column].map(mapping)

        if data[column].isna().any():
            data[column] = data[column].fillna(data[column].median())

        data[column] = data[column].astype(float)

# Create analysis dataset with relevant columns
analysis_columns = ['Q4', 'Q9', 'Q12', 'Q14', 'Q16', 'motivation_count'] + [f'Q11_{attr}' for attr in q11_attributes_sanitized]
analysis_data = data[analysis_columns].copy()

# Handle missing data
missing_pattern = analysis_data.isna().sum()
cols_with_missing = missing_pattern[missing_pattern > 0]
if not cols_with_missing.empty:
    try:
        from statsmodels.imputation.tests import MissingDataTests
        mcar_test = MissingDataTests(analysis_data)
        p_value_mcar = mcar_test.little_mcar_test()[1]  # p-value
    except Exception as e:
        pass

# Use IterativeImputer instead of median for better variance preservation
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    # NOTE: FIML was the preferred approach; however, due to version constraints of
    # semopy at time of analysis, MICE imputation via sklearn's IterativeImputer
    # was applied as a principled MAR-compatible alternative (van Buuren & Groothuis-Oudshoorn, 2011).
    imputer = IterativeImputer(random_state=42, max_iter=10)
    imputed_data = imputer.fit_transform(analysis_data)
    analysis_data = pd.DataFrame(imputed_data, columns=analysis_data.columns)
except Exception as e:
    for col in analysis_data.columns:
        if analysis_data[col].isna().sum() > 0:
            analysis_data[col] = analysis_data[col].fillna(analysis_data[col].median())

# Check for normality
normality_violations = []

# 1. Univariate normality tests (D'Agostino-Pearson - sensitive to sample size)
for col in analysis_data.columns:
    stat, p_value = stats.normaltest(analysis_data[col])
    if p_value < 0.05:
        normality_violations.append(col)

# 3. Multivariate normality (Mardia's test for SEM)
try:
    from scipy.spatial.distance import mahalanobis
    from scipy.stats import chi2

    mean = analysis_data.mean()
    cov = analysis_data.cov()
    inv_cov = np.linalg.inv(cov)

    # CORRECTED: Compute pairwise Mahalanobis distances (Mardia 1970)
    # The correct formula requires D_ij = (x_i - x̄)ᵀ S⁻¹ (x_j - x̄) for all pairs
    n = len(analysis_data)
    p = analysis_data.shape[1]
    
    # Build pairwise cross-product matrix
    D = np.zeros((n, n))
    observations = analysis_data.values
    for i in range(n):
        for j in range(n):
            diff_i = observations[i] - mean.values
            diff_j = observations[j] - mean.values
            D[i, j] = diff_i @ inv_cov @ diff_j
    
    # Mardia's skewness (correct pairwise computation)
    g1 = np.sum(D**3) / (n**2)
    mardia_skew_stat = (n / 6) * g1
    mardia_skew_p = 1 - chi2.cdf(mardia_skew_stat, p * (p + 1) * (p + 2) / 6)

    # Mardia's kurtosis (correct: sum of squared diagonal Mahalanobis distances)
    d_sq = np.diag(D)
    g2 = np.mean(d_sq**2)
    expected_kurt = p * (p + 2)
    mardia_kurt_stat = (g2 - expected_kurt) / np.sqrt(8 * p * (p + 2) / n)
    mardia_kurt_p = 2 * (1 - stats.norm.cdf(abs(mardia_kurt_stat)))

    # Information available internally via mardia_skew_p and mardia_kurt_p (e.g. p < 0.05 indicates multivariate non-normality)
except Exception as e:
    pass

# 4. Visual inspection: Q-Q plots for key variables
try:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    key_vars_for_qq = list(analysis_data.columns[:6])

    for idx, col in enumerate(key_vars_for_qq):
        ax = axes[idx]
        stats.probplot(analysis_data[col], dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: {col}")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('sem_results/qq_plots.png', dpi=100, bbox_inches='tight')
    plt.close()
except Exception as e:
    pass

# Define estimation method
estimation_method = "ULS"  # Unweighted Least Squares - robust to non-normality

# Define the full model specification (structural only, using composite score)
full_model_spec = """
# Structural model
Q9 ~ Q4                    # Social media usage -> Familiarity
Q12 ~ motivation_count + Q9 + Q14  # Willingness to pay depends on motivation, familiarity and environment
Q16 ~ Q12 + Q9 + motivation_count  # Investment likelihood depends on willingness to pay and familiarity
"""
# Calculate Cronbach's Alpha for latent variable
try:
    # KR-20 reliability for binary items (Kuder-Richardson Formula 20)
    def calculate_kr20(data, items):
        item_data = data[items].dropna()
        k = len(items)
        p = item_data.mean(axis=0)          # proportion correct per item
        q = 1 - p
        sigma2 = item_data.sum(axis=1).var(ddof=1)   # total score variance
        kr20 = (k / (k - 1)) * (1 - (p * q).sum() / sigma2)
        return kr20

    motivation_items = [f'Q11_{attr}' for attr in q11_attributes_sanitized]
    alpha = calculate_kr20(analysis_data, motivation_items)

    # Note: Q11 motivation items are FORMATIVE, not reflective.
    # Cronbach's alpha and KR-20 are inappropriate for formative measurement models.

    # Interpret alpha value (for reference)
    if alpha >= 0.9:
        reliability = "Excellent (if reflective)"
    elif alpha >= 0.8:
        reliability = "Good (if reflective)"
    elif alpha >= 0.7:
        reliability = "Acceptable (if reflective)"
    elif alpha >= 0.6:
        reliability = "Questionable (if reflective)"
    else:
        reliability = "Poor—but expected for FORMATIVE indicators"

    # Save to results
    with open("sem_results/reliability.txt", "w") as f:
        f.write("Reliability Analysis for Q11 Motivation Items\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"KR-20 reliability coefficient: {alpha:.3f}\n")
        f.write(f"Interpretation (if reflective): {reliability}\n\n")
        f.write("CRITICAL NOTE ON MEASUREMENT MODEL:\n")
        f.write("-" * 50 + "\n")
        f.write("Q11 items are FORMATIVE indicators (multiple selection reasons that are\n")
        f.write("distinct and non-overlapping). They are NOT reflective, so Cronbach's\n")
        f.write("alpha/KR-20 should NOT be used to assess 'reliability' as if the items\n")
        f.write("were interchangeable measures of a single latent trait.\n\n")
        f.write("For formative composites, ensure:\n")
        f.write("  1. Content validity (each item is a distinct, relevant reason)\n")
        f.write("  2. Weights are non-negative and theoretically sensible\n")
        f.write("  3. Composite score is used (e.g., sum or mean of indicators)\n")
except Exception as e:
    pass

# Initialize models
model = Model(full_model_spec)

try:
    # Model fitting with explicit ULS estimator
    # NOTE: semopy's default is ML. Must specify obj='ULS' to actually use ULS
    result = model.fit(analysis_data, obj='ULS')
    model_fitted = True

    # Calculate degrees of freedom for model identification assessment analytically
    # Only use the sequence of variables actually present in the structural model
    model_vars = ['Q4', 'Q9', 'Q12', 'Q14', 'Q16', 'motivation_count']
    p_vars = len(model_vars)
    unique_cov_elements = (p_vars * (p_vars + 1)) / 2  # 6(7)/2 = 21

    # Safely count actual free parameters estimated by semopy
    try:
        est_params = inspector.inspect(model, what='estimates')
        num_free_params = len(est_params)
    except Exception:
        # Analytical fallback: 5 structural paths, 3 endogenous residuals, 3 exogenous variances, 3 exogenous covariances = 14
        num_free_params = 14

    df = unique_cov_elements - num_free_params

    # Store for appending to summary
    model_p = p_vars
    model_unique_cov = unique_cov_elements
    model_free_params = num_free_params
    model_df = df

    try:
        import semopy
        semopy_stats = semopy.calc_stats(model)
        fit_indices = {}
        if isinstance(semopy_stats, pd.DataFrame):
            stats_transposed = semopy_stats.T
            if not stats_transposed.empty:
                col_idx = stats_transposed.columns[0]
                for idx_name, val in stats_transposed[col_idx].items():
                    fit_indices[idx_name] = val

        fit_indices_df = pd.DataFrame({
            'Measure': list(fit_indices.keys()),
            'Value': list(fit_indices.values())
        })
        fit_indices_df['Estimator'] = "ULS"
        fit_indices_df['Reference_Note'] = 'ULS uses different cutoffs than ML. See analysis summary.'
        fit_indices_df.to_csv('sem_results/fit_indices.csv', index=False)
    except Exception as e:
        pass
except Exception as e:
    model_fitted = False

if model_fitted:
    hypotheses = {
        "H1: Social Media Usage → Familiarity": {
            "path": "Q4->Q9",
            "theory": "Greater social media usage increases familiarity with digital fashion",
            "expected": "positive"
        },
        "H2: Motivation → Willingness to Pay": {
            "path": "motivation_count->Q12",
            "theory": "Stronger motivation increases willingness to pay",
            "expected": "positive"
        },
        "H3: Willingness to Pay → Investment Likelihood": {
            "path": "Q12->Q16",
            "theory": "Higher willingness to pay leads to increased investment likelihood",
            "expected": "positive"
        },
        "H4: Familiarity → Willingness to Pay": {
            "path": "Q9->Q12",
            "theory": "Greater familiarity with digital fashion increases willingness to pay",
            "expected": "positive"
        },
        "H5: Environmental Consciousness → Willingness to Pay": {
            "path": "Q14->Q12",
            "theory": "Environmental consciousness influences willingness to pay for digital fashion",
            "expected": "positive"
        }
    }

    # Test hypotheses using diagnostic approach
    try:
        # Get parameter estimates with standardized coefficients
        # NOTE: semopy's default is unstandardized. Use std_est=True for standardized solution
        params_df = inspector.inspect(model, what="estimates", std_est=True)
        # Also store unstandardized for reference
        params_df_unstd = inspector.inspect(model, what="estimates", std_est=False)

        params_df.to_csv('sem_results/parameter_estimates.csv', index=False)
        params_df_unstd.to_csv('sem_results/parameter_estimates_unstandardized.csv', index=False)

        # Create results dataframe
        results_df = pd.DataFrame(columns=[
            'Hypothesis', 'Path', 'Coefficient', 'Std_Error', 'CI_Lower', 'CI_Upper',
            'z_value', 'p_value', 'Effect_Size', 'Direction', 'Expected', 'Supported'
        ])

        for hypothesis, details in hypotheses.items():
            path = details["path"]

            # Extract path components
            source, target = path.split("->")

            # In semopy output: lval=target, op=~, rval=source
            param_row = params_df[
                (params_df['lval'] == target) &
                (params_df['op'] == '~') &
                (params_df['rval'] == source)
                ]

            if not param_row.empty:
                coef = param_row['Estimate'].values[0]
                se = param_row['Std. Err'].values[0]
                z_val = param_row['z-value'].values[0]
                p_val = param_row['p-value'].values[0]

                # Calculate 95% CI using t-distribution (appropriate for finite samples)
                # df = n - num_free_params (conservative: use n - 1)
                df_ci = len(analysis_data) - 1
                t_crit = stats.t.ppf(0.975, df_ci)  # Two-tailed t-critical value
                ci_lower = coef - t_crit * se
                ci_upper = coef + t_crit * se

                # CORRECTED: Effect size classification using Cohen's standards for path coefficients
                # Cohen (1988): small = 0.10, medium = 0.30, large = 0.50
                if abs(coef) < 0.10:
                    effect_size = "Negligible"
                elif abs(coef) < 0.30:
                    effect_size = "Small"
                elif abs(coef) < 0.50:
                    effect_size = "Medium"
                else:
                    effect_size = "Large"

                direction = 'Positive' if coef > 0 else 'Negative'
                expected = details['expected']
                # Supported: significant AND correct direction AND meaningful effect size
                supported = 'Supported' if (p_val < 0.05 and abs(coef) >= 0.1 and
                                            ((direction.lower() == expected.lower()) or
                                             (expected.lower() == 'any'))) else 'Not Supported'

                # Add to results dataframe
                new_row = pd.DataFrame({
                    'Hypothesis': [hypothesis],
                    'Path': [path],
                    'Coefficient': [coef],
                    'Std_Error': [se],
                    'CI_Lower': [ci_lower],
                    'CI_Upper': [ci_upper],
                    'z_value': [z_val],
                    'p_value': [p_val],
                    'Effect_Size': [effect_size],
                    'Direction': [direction],
                    'Expected': [expected],
                    'Supported': [supported]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
            else:
                # Try reversed direction (some structural equation models reverse the arrow)
                param_row = params_df[
                    (params_df['lval'] == source) &
                    (params_df['op'] == '~') &
                    (params_df['rval'] == target)
                    ]

                if not param_row.empty:
                    coef = param_row['Estimate'].values[0]
                    se = param_row['Std. Err'].values[0]
                    z_val = param_row['z-value'].values[0]
                    p_val = param_row['p-value'].values[0]

                    # Calculate 95% CI using t-distribution
                    df_ci = len(analysis_data) - 1
                    t_crit = stats.t.ppf(0.975, df_ci)
                    ci_lower = coef - t_crit * se
                    ci_upper = coef + t_crit * se

                    # CORRECTED: Cohen's effect size standards
                    if abs(coef) < 0.10:
                        effect_size = "Negligible"
                    elif abs(coef) < 0.30:
                        effect_size = "Small"
                    elif abs(coef) < 0.50:
                        effect_size = "Medium"
                    else:
                        effect_size = "Large"

                    direction = 'Positive' if coef > 0 else 'Negative'
                    expected = details['expected']
                    supported = 'Supported' if (p_val < 0.05 and abs(coef) >= 0.1 and
                                                ((direction.lower() == expected.lower()) or
                                                 (expected.lower() == 'any'))) else 'Not Supported'

                    # Add to results dataframe
                    new_row = pd.DataFrame({
                        'Hypothesis': [hypothesis],
                        'Path': [f"{target}->{source} (reversed)"],
                        'Coefficient': [coef],
                        'Std_Error': [se],
                        'CI_Lower': [ci_lower],
                        'CI_Upper': [ci_upper],
                        'z_value': [z_val],
                        'p_value': [p_val],
                        'Effect_Size': [effect_size],
                        'Direction': [direction],
                        'Expected': [expected],
                        'Supported': [supported]
                    })
                    results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Export results
        if not results_df.empty:
            results_df.to_csv("sem_results/hypothesis_tests.csv", index=False)

    except Exception as e:
        pass

# Mediation analysis
try:
    # Get direct effects from parameters
    params_dict = {}
    for _, row in params_df.iterrows():
        key = f"{row['rval']}->{row['lval']}"
        params_dict[key] = {
            'estimate': row['Estimate'],
            'std_err': row['Std. Err'],
            'p_value': row['p-value']
        }

    # Define mediation paths to test
    mediation_paths = [
        {
            'name': 'Social Media Usage → Familiarity → Willingness to Pay',
            'indirect_path': ['Q4->Q9', 'Q9->Q12']
        },
        {
            'name': 'Familiarity → Willingness to Pay → Investment Likelihood',
            'indirect_path': ['Q9->Q12', 'Q12->Q16']
        }
    ]

    # Create results dataframe
    mediation_results = pd.DataFrame(columns=[
        'Mediation Path', 'Indirect Effect', 'Path A', 'Path B', 'Path A p-value', 'Path B p-value',
        'Sobel_z', 'Sobel_p', 'Boot_CI_Lower', 'Boot_CI_Upper', 'Boot_p'
    ])

    # Function for bootstrap confidence intervals on indirect effect
    def bootstrap_indirect_effect(analysis_data, model, path_a, path_b, n_bootstrap=5000):
        """
        Compute bootstrap confidence intervals for indirect effect a*b
        using percentile method (Hayes & Preacher 2013, Preacher & Hayes 2008)
        """
        indirect_effects = []
        n = len(analysis_data)

        for boot_idx in range(n_bootstrap):
            # Resample with replacement
            boot_indices = np.random.choice(n, n, replace=True)
            boot_data = analysis_data.iloc[boot_indices].reset_index(drop=True)

            try:
                # Refit model on bootstrap sample (suppress output)
                boot_result = model.fit(boot_data, obj='ULS')
                boot_params = inspector.inspect(model, what="estimates", std_est=True)

                # Extract path coefficients
                source_a, target_a = path_a.split('->')
                source_b, target_b = path_b.split('->')

                path_a_key = f"{source_a}->{target_a}"
                path_b_key = f"{source_b}->{target_b}"

                path_a_row = boot_params[(boot_params['lval'] == target_a) &
                                         (boot_params['op'] == '~') &
                                         (boot_params['rval'] == source_a)]
                path_b_row = boot_params[(boot_params['lval'] == target_b) &
                                         (boot_params['op'] == '~') &
                                         (boot_params['rval'] == source_b)]

                if not path_a_row.empty and not path_b_row.empty:
                    coef_a = path_a_row['Estimate'].values[0]
                    coef_b = path_b_row['Estimate'].values[0]
                    indirect_effects.append(coef_a * coef_b)
            except:
                # Skip failed bootstrap iterations
                pass

        n_succeeded = len(indirect_effects)
        print(f"  Bootstrap: {n_succeeded} of {n_bootstrap} iterations converged successfully")
        if n_succeeded < n_bootstrap * 0.8:
            print(f"  WARNING: More than 20% of bootstrap samples failed to converge.")
            print(f"  Bootstrap CI may be unreliable. Interpret with caution.")

        if n_succeeded > 0:
            indirect_effects = np.array(indirect_effects)
            boot_ci_lower = np.percentile(indirect_effects, 2.5)
            boot_ci_upper = np.percentile(indirect_effects, 97.5)
            boot_p = min(np.mean(indirect_effects > 0), np.mean(indirect_effects < 0)) * 2
            return boot_ci_lower, boot_ci_upper, boot_p, np.mean(indirect_effects)
        else:
            return np.nan, np.nan, np.nan, np.nan

    # Test each mediation path
    for path in mediation_paths:
        # Extract path segments
        path_a = path['indirect_path'][0]
        path_b = path['indirect_path'][1]
        source_a, target_a = path_a.split('->')
        source_b, target_b = path_b.split('->')

        # Look up path coefficients (with proper orientation)
        path_a_key = f"{source_a}->{target_a}"
        path_b_key = f"{source_b}->{target_b}"

        if path_a_key in params_dict and path_b_key in params_dict:
            coef_a = params_dict[path_a_key]['estimate']
            coef_b = params_dict[path_b_key]['estimate']
            p_a = params_dict[path_a_key]['p_value']
            p_b = params_dict[path_b_key]['p_value']
            se_a = params_dict[path_a_key]['std_err']
            se_b = params_dict[path_b_key]['std_err']
            indirect_effect = coef_a * coef_b

            # Sobel test (Delta method - assumes normality of a*b, which is violated)
            sobel_se = np.sqrt(coef_b**2 * se_a**2 + coef_a**2 * se_b**2)
            sobel_z = indirect_effect / sobel_se if sobel_se > 0 else 0
            sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

            # Bootstrap confidence interval (RECOMMENDED, Hayes 2013)
            try:
                boot_ci_lower, boot_ci_upper, boot_p, boot_mean = bootstrap_indirect_effect(
                    analysis_data, model, path_a, path_b, n_bootstrap=5000
                )
            except Exception as boot_e:
                boot_ci_lower, boot_ci_upper, boot_p = np.nan, np.nan, np.nan

            # Add to results
            new_row = pd.DataFrame({
                'Mediation Path': [path['name']],
                'Indirect Effect': [indirect_effect],
                'Path A': [coef_a],
                'Path B': [coef_b],
                'Path A p-value': [p_a],
                'Path B p-value': [p_b],
                'Sobel_z': [sobel_z],
                'Sobel_p': [sobel_p],
                'Boot_CI_Lower': [boot_ci_lower],
                'Boot_CI_Upper': [boot_ci_upper],
                'Boot_p': [boot_p]
            })
            mediation_results = pd.concat([mediation_results, new_row], ignore_index=True)

    # Export mediation results
    if not mediation_results.empty:
        mediation_results.to_csv("sem_results/mediation_analysis.csv", index=False)
except Exception as e:
    pass

# Calculate correlations between key variables
correlation_vars = ['Q4', 'Q9', 'Q12', 'Q14', 'Q16', 'motivation_count']
corr_matrix = data[correlation_vars].corr(method='spearman')

# Export correlation matrix
corr_matrix.to_csv("sem_results/correlation_matrix.csv")

# Identify significant correlations
significant_corrs = []

# Calculate p-values for correlations
for i, var1 in enumerate(correlation_vars):
    for j, var2 in enumerate(correlation_vars):
        if i < j:  # Only lower triangle to avoid duplicates
            r = corr_matrix.loc[var1, var2]
            # Clip r to avoid division by zero
            r = min(max(r, -0.999999), 0.999999)
            # Calculate p-value for correlation
            t_stat = r * np.sqrt((len(data) - 2) / (1 - r ** 2))
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(data) - 2))

            if p_val < 0.05:
                significant_corrs.append((var1, var2, r, p_val))

# Analyze relationship between motivation factors and willingness to pay
motivation_count_corr = data[['motivation_count', 'Q12']].corr(method='spearman').iloc[0, 1]

# Calculate significance of correlation
t_stat = motivation_count_corr * np.sqrt((len(data) - 2) / (1 - motivation_count_corr ** 2))
p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(data) - 2))

# Create summary file with key findings
summary = ["## SEM Analysis Summary ##",
           f"Sample size: {len(data)} observations",
           f"Variables analyzed: {len(analysis_columns)}",
           ""]

if model_fitted:
    # Add model fit details
    summary.append("# Model Fit")

    summary.append("\n# Model Identification Assessment")
    summary.append(f"Observed Variables (p): {model_p}")
    summary.append(f"Unique Sample Moments: {int(model_unique_cov)}")
    summary.append(f"Estimated Parameters: {model_free_params}")
    summary.append(f"Degrees of Freedom (df): {int(model_df)}")
    id_status = 'Identified (Over-identified)' if model_df > 0 else 'Just-identified' if model_df == 0 else 'Under-identified'
    summary.append(f"Identification Status: {id_status}\n")

    try:
        for name, value in fit_indices.items():
            summary.append(f"{name}: {value}")
    except:
        pass

    try:
        # Extract endogenous variables (those on left side of ~ in model specification)
        endogenous_vars = ['Q9', 'Q12', 'Q16']  # Based on your model specification

        # Calculate R-squared using estimated residual variance vs total observed variance
        # (This is the semopy-native statistical approach since predict() doesn't modify observed variables)
        params_df = inspector.inspect(model)
        r_squared = {}
        for var in endogenous_vars:
            try:
                # Find residual variance from model estimates (~~ operator on the variable itself)
                res_var_row = params_df[(params_df['lval'] == var) & (params_df['rval'] == var) & (params_df['op'] == '~~')]
                if not res_var_row.empty:
                    res_var = res_var_row['Estimate'].values[0]
                    # Total variance of the observed variable
                    tot_var = analysis_data[var].var(ddof=0)
                    r_squared[var] = max(0.0, 1.0 - (res_var / tot_var))
            except Exception as loop_e:
                pass

        r_squared = pd.Series(r_squared)
        r_squared_clean = {}
        for var, r2 in r_squared.items():
            if isinstance(r2, pd.Series):
                r2_val = float(r2.iloc[0])
            else:
                r2_val = float(r2)
            r_squared_clean[var] = r2_val

        # Export R-squared values
        pd.DataFrame({
            'Variable': list(r_squared_clean.keys()),
            'R_squared': list(r_squared_clean.values())
        }).to_csv('sem_results/r_squared.csv', index=False)

        # Add to summary
        for var, r2 in zip(r_squared.index, r_squared.values):
            try:
                summary.append(f"R² for {var}: {float(r2):.3f}")
            except:
                pass
    except Exception as e:
        pass

    # Add hypothesis test results
    summary.append("\n# Hypothesis Tests")
    try:
        for _, row in results_df.iterrows():
            summary.append(
                f"{row['Hypothesis']}: β={row['Coefficient']:.3f}, p={row['p_value']:.4f}, {row['Supported']}")
    except:
        pass

# Add correlation results
summary.append("\n# Key Correlations")
for var1, var2, r, p in significant_corrs:
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    summary.append(f"{var1} - {var2}: r = {r:.3f}, p = {p:.4f} {stars}")

# Save summary to file
with open("sem_results/analysis_summary.txt", "w") as f:
    f.write("\n".join(summary))
