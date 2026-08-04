import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import networkx as nx
from scipy import stats
import os
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titlepad'] = 14
plt.rcParams['axes.labelpad'] = 8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['legend.frameon'] = False
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#FAFAF9'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#E0DED8'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.8

output_dir = 'publication_figures'
relationship_dir = 'relationship_visualizations'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(relationship_dir, exist_ok=True)

C_PURPLE  = '#7F77DD'
C_TEAL    = '#1D9E75'
C_CORAL   = '#D85A30'
C_BLUE    = '#378ADD'
C_AMBER   = '#EF9F27'
C_PINK    = '#D4537E'
C_GREEN   = '#639922'
C_GRAY    = '#888780'
C_RED     = '#E24B4A'

PALETTE_9 = [C_PURPLE, C_TEAL, C_CORAL, C_BLUE, C_AMBER, C_PINK, C_GREEN, C_GRAY, C_RED]
main_palette      = sns.color_palette(PALETTE_9)
categorical_palette = [C_TEAL, C_CORAL, C_PURPLE, C_BLUE, C_AMBER, C_PINK]
diverging_palette = sns.color_palette("RdBu_r", 9)
sequential_palette = [
    '#E1F5EE', '#9FE1CB', '#5DCAA5', '#1D9E75',
    '#0F6E56', '#085041', '#04342C'
]

data = pd.read_csv('/Users/suhasishbasak/Consumer Survey - SEM/cleaned_data.csv')

# Safe file loading with better error reporting
def safe_load_csv(filepath, description):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"{description} file not found")
        return None
    except Exception as e:
        print(f"Error loading {description}: {str(e)}")
        return None

results = safe_load_csv('sem_results/hypothesis_tests.csv', "Hypothesis tests")
fit_indices = safe_load_csv('sem_results/fit_indices.csv', "Fit indices")
r_squared = safe_load_csv('sem_results/r_squared.csv', "R-squared")
correlations = safe_load_csv('sem_results/correlation_matrix.csv', "Correlation matrix")

key_variables = {
    'Q4': 'Instagram Engagement',
    'Q9': 'Digital Fashion Awareness',
    'Q11': 'Purchase Motivation',
    'Q12': 'Willingness to Pay',
    'Q14': 'Environmental Awareness',
    'Q16': 'Willingness to Invest'
}

data_numeric = data.copy()

data['Q4_original'] = data['Q4']
data['Q9_original'] = data['Q9']
data['Q12_original'] = data['Q12']

engagement_map = {
    '5 + hours': 5,
    '4 - 5': 4,
    '2 - 3': 3,
    '0 - 1': 2,
    'Not at all': 1
}
data_numeric['Q4_numeric'] = data['Q4'].map(engagement_map)

awareness_map = {
    'Yes': 3,
    'Not sure': 2,
    'Not Sure': 2,
    'No': 1
}
data_numeric['Q9_numeric'] = data['Q9'].map(awareness_map)

price_map = {
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
data_numeric['Q12_numeric'] = data['Q12'].map(price_map) if data['Q12'].dtype == 'object' else data['Q12']

q11_attributes = ['Design', 'Uniqueness', 'Reputation', 'Curiosity',
                 'Social Media Influence', 'Price', 'Environment',
                 'Customisation', 'Utility']

q11_attributes_sanitized = [attr.replace(' ', '_').lower() for attr in q11_attributes]

# Create binary columns for each motivation attribute
for attr, attr_clean in zip(q11_attributes, q11_attributes_sanitized):
    data[f'Q11_{attr_clean}'] = data['Q11'].str.contains(attr).astype(int)
    data_numeric[f'Q11_{attr_clean}'] = data[f'Q11_{attr_clean}']

data['motivation_count'] = data[[f'Q11_{attr_clean}' for attr_clean in q11_attributes_sanitized]].sum(axis=1)
data_numeric['motivation_count'] = data['motivation_count']

data['Motivation_Score'] = data[[f'Q11_{attr_clean}' for attr_clean in q11_attributes_sanitized]].mean(axis=1)
data_numeric['Motivation_Score'] = data['Motivation_Score']

data['Instagram Engagement'] = data['Q4'].map({
    '5 + hours': '5+ Hours',
    '4 - 5': '4-5 Hours',
    '2 - 3': '2-3 Hours',
    '0 - 1': '0-1 Hours',
    'Not at all': 'Not Engaged'
})

data['Digital Fashion Awareness'] = data['Q9'].map({
    'Yes': 'Aware',
    'Not sure': 'Uncertain',
    'No': 'Unaware'
})

data['Willingness_to_Pay'] = data['Q12'].map({
    'Above $200/ £162/ ₹17,252': 'Above $200',
    '($101 - $200)/ (£82 - £162)/ (₹7073 - ₹17,252)': '$101-$200',
    '($101 - $200)/ (£82 -  £162)/ (₹7073 - ₹17,252)': '$101-$200',
    '($51 - $100)/ (£41 - £81)/ (₹3535 - ₹8626)': '$51-$100',
    '($51 - $100)/ (£41 -  £81)/ (₹3535 - ₹8626)': '$51-$100',
    '($21 - $50)/ (£17 - £40)/ (₹1810 - ₹4310)': '$21-$50',
    '($21 - $50)/ (£17 -  £40)/ (₹1810 - ₹4310)': '$21-$50',
    '($5 - $20)/ (£4 - £16)/ (₹430 - ₹1725)': '$5-$20',
    '($5 - $20)/ (£4 -  £16)/ (₹430 - ₹1725)': '$5-$20',
    'I would rather not buy': 'Not buying'
})

# Utility function to safely place text in plots
def safe_text(ax, x, y, text, **kwargs):
    """Place text at coordinates (x,y) only if both are finite values"""
    if np.isfinite(x) and np.isfinite(y):
        ax.text(x, y, text, **kwargs)
    else:
        print(f"Warning: Cannot place text at ({x}, {y})")

def visualize_correlation_matrix(corr_matrix, output_path=None):
    """Create a publication-quality correlation matrix heatmap

    Note: This function accepts pre-computed correlation matrices.
    For PhD thesis rigor: Spearman's ρ (rank-based) is used instead of Pearson's r
    because variables include ordinal (Q9, Q12, Q14, Q16) and binary indicators (Q11_*).
    Spearman's ρ is more appropriate for non-parametric data.
    """
    plt.figure(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    cmap = LinearSegmentedColormap.from_list('correlation',
        ['#D85A30', '#F09595', '#F1EFE8', '#85B7EB', '#185FA5'])

    ax = sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                    vmin=-1, vmax=1, square=True, linewidths=0.5,
                    cbar_kws={"shrink": .8})

    col_mapping = {'Q4_numeric': 'Instagram Engagement',
                   'Q9_numeric': 'Digital Fashion Awareness',
                   'Q12_numeric': 'Willingness to Pay',
                   'Q14': 'Environmental Awareness',
                   'Q16': 'Willingness to Invest',
                   'motivation_count': 'Motivation Count',
                   'Motivation_Score': 'Motivation Score'}

    labels = [col_mapping.get(col, col) for col in corr_matrix.columns]
    ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(labels, rotation=0, va='center')

    plt.title('Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix visualization saved to {output_path}")

    plt.close()

def visualize_hypothesis_tests(results_df, output_path=None):
    """Create a visualization of hypothesis test results with effect sizes and CIs"""
    if results_df is None:
        print("Cannot visualize hypothesis tests: missing results data")
        return

    # Sort by hypothesis number
    results_df['Hypothesis_Num'] = results_df['Hypothesis'].str.extract(r'H(\d+)').astype(int)
    results_df = results_df.sort_values('Hypothesis_Num')

    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.barh(
        results_df['Hypothesis'], results_df['Coefficient'],
        color=[C_TEAL if s == 'Supported' else C_CORAL for s in results_df['Supported']],
        edgecolor='white', linewidth=0.6, height=0.55
    )

    if 'CI_Lower' in results_df.columns and 'CI_Upper' in results_df.columns:
        x_err = [results_df['Coefficient'] - results_df['CI_Lower'],
                 results_df['CI_Upper'] - results_df['Coefficient']]
        ax.errorbar(results_df['Coefficient'], range(len(results_df)),
                   xerr=x_err, fmt='none', ecolor='black', capsize=5, capthick=2, alpha=0.5)

    for i, bar in enumerate(bars):
        row = results_df.iloc[i]
        width = row['Coefficient']
        label_x_pos = width + 0.02 if width >= 0 else width - 0.08

        if 'Effect_Size' in results_df.columns:
            effect_label = f"{width:.3f}\n({row['Effect_Size']})"
        else:
            effect_label = f"{width:.3f}"

        safe_text(ax, label_x_pos, bar.get_y() + bar.get_height() / 2,
                 effect_label, va='center', fontsize=9)

    ax.axvline(x=0, color='black', linestyle='-', alpha=0.7, linewidth=0.8)

    ax.set_xlabel('Standardized Path Coefficient (β) with 95% CI', fontsize=12)
    ax.set_title('Hypothesis Testing Results with Effect Sizes', fontsize=16, pad=20)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_TEAL,   markersize=10, label='Supported'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_CORAL, markersize=10, label='Not supported')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Hypothesis test visualization saved to {output_path}")

    plt.close()

# Q9_numeric (ordinal), Q12_numeric (ordinal), and Motivation_Score (derived from binary Q11 indicators)
# are more appropriately analyzed with Spearman's ρ than Pearson's r
corr_columns = ['Q4_numeric', 'Q9_numeric', 'Q12_numeric', 'Q14', 'Q16', 'Motivation_Score']
corr_matrix_df = data_numeric[corr_columns].corr(method='spearman')
visualize_correlation_matrix(corr_matrix_df, output_path=f'{output_dir}/correlation_matrix.png')

# Call the hypothesis test visualization function
if results is not None:
    try:
        visualize_hypothesis_tests(results, output_path=f'{output_dir}/hypothesis_tests.png')
    except Exception as e:
        print(f"Error generating hypothesis test visualization: {str(e)}")
else:
    print("Skipping hypothesis test visualization due to missing data")

def visualize_path_coefficients(results_df=None, output_path=None):
    """Create a visualization of path coefficients with significance levels"""
    if results_df is None:
        print("Cannot visualize path coefficients: missing results data")
        return

    plt.figure(figsize=(12, 8))

    results_df = results_df.sort_values(by='Coefficient', key=abs, ascending=False)

    bars = plt.barh(
        y=results_df['Path'] if 'Path' in results_df.columns else results_df['Hypothesis'],
        width=results_df['Coefficient'],
        color=[C_PURPLE if p < 0.05 else C_GRAY for p in results_df['p_value']],
        edgecolor='white', linewidth=0.6, height=0.55
    )

    for i, bar in enumerate(bars):
        row = results_df.iloc[i]
        width = bar.get_width()
        stars = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row[
                                                                                                   'p_value'] < 0.05 else ''

        if np.isfinite(width):
            plt.text(
                width + 0.02 if width >= 0 else width - 0.08,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f} {stars}",
                va='center',
                fontweight='bold' if row['p_value'] < 0.05 else 'normal'
            )

    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.xlabel('Standardized Path Coefficient (β)')
    plt.title('Path Coefficients with Significance Levels')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.figtext(0.01, 0.01, "* p < 0.05, ** p < 0.01, *** p < 0.001", ha='left', fontsize=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Path coefficient visualization saved to {output_path}")
    plt.close()

def visualize_mediation_effects(output_path=None):
    """Create a visualization of mediation effects with Bootstrap CI and Sobel test results"""
    try:
        mediation_df = pd.read_csv('sem_results/mediation_analysis.csv')

        if mediation_df.empty or not np.isfinite(mediation_df['Indirect Effect']).all():
            print("No valid mediation effects to visualize")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.barh(mediation_df['Mediation Path'],
                        mediation_df['Indirect Effect'],
                        color=main_palette[3],
                        edgecolor='white', linewidth=0.6, height=0.55)

        # Iterate and add error bars for Bootsrap CI if available
        for i, bar in enumerate(bars):
            width = bar.get_width()
            y_pos = bar.get_y() + bar.get_height() / 2

            if np.isfinite(width):
                has_boot = 'Boot_CI_Lower' in mediation_df.columns and 'Boot_CI_Upper' in mediation_df.columns

                if has_boot:
                    ci_lower = mediation_df.iloc[i]['Boot_CI_Lower']
                    ci_upper = mediation_df.iloc[i]['Boot_CI_Upper']
                    if np.isfinite(ci_lower) and np.isfinite(ci_upper):
                        # Add error bar
                        ax.errorbar(width, y_pos, xerr=[[width - ci_lower], [ci_upper - width]],
                                    fmt='none', ecolor='black', capsize=4, capthick=1, elinewidth=1.2)

                # Format text
                if 'Boot_p' in mediation_df.columns and np.isfinite(mediation_df.iloc[i]['Boot_p']):
                    p_val = mediation_df.iloc[i]['Boot_p']
                    stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                    label_text = f"{width:.3f} {stars}"
                elif 'Sobel_p' in mediation_df.columns and np.isfinite(mediation_df.iloc[i]['Sobel_p']):
                    sobel_p = mediation_df.iloc[i]['Sobel_p']
                    stars = '***' if sobel_p < 0.001 else '**' if sobel_p < 0.01 else '*' if sobel_p < 0.05 else 'ns'
                    label_text = f"{width:.3f} (Sobel) {stars}"
                else:
                    label_text = f"{width:.3f}"

                # Adjust text position to skip error bar
                text_x = ci_upper + 0.01 if has_boot and np.isfinite(ci_upper) else width + 0.005
                plt.text(text_x, y_pos, label_text, va='center', fontsize=9)

        ax.set_xlabel('Indirect Effect Size (with 95% Bootstrap CI)')
        ax.set_title('Mediation Analysis: Indirect Effects (Bootstrap CI & Significance)')
        ax.grid(axis='x', linestyle='--', alpha=0.3)

        plt.figtext(0.01, 0.01, "* p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant",
                   ha='left', fontsize=9)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Mediation effects visualization saved to {output_path}")
        plt.close()

    except FileNotFoundError:
        print("Mediation analysis file not found - skipping mediation visualization")
    except Exception as e:
        print(f"Error visualizing mediation effects: {str(e)}")

def visualize_fit_indices(fit_indices_df=None, output_path=None):
    """Create a visualization of fit indices with estimator information"""
    if fit_indices_df is None or fit_indices_df.empty:
        print("Cannot visualize fit indices: missing data")
        return

    try:
        estimator = fit_indices_df.get('Estimator', ['ULS']).iloc[0] if 'Estimator' in fit_indices_df.columns else 'ULS'

        if 'Measure' in fit_indices_df.columns:
            indices = fit_indices_df['Measure'].values
        elif 'Index' in fit_indices_df.columns:
            indices = fit_indices_df['Index'].values
        else:
            indices = fit_indices_df.iloc[:, 0].values if len(fit_indices_df.columns) > 0 else None

        if indices is None:
            print("Cannot visualize fit indices: no measure column found")
            return

        values = fit_indices_df['Value'].values if 'Value' in fit_indices_df.columns else fit_indices_df.iloc[:, 1].values

        plt.figure(figsize=(12, 6))

        colors = []
        clean_values = []
        for measure, value in zip(indices, values):
            if isinstance(value, str):
                colors.append(C_GRAY)
                clean_values.append(0.0)
            else:
                try:
                    val = float(value)
                    if not np.isfinite(val):
                        colors.append(C_GRAY)
                        clean_values.append(0.0)
                    elif any(x in str(measure) for x in ['CFI', 'GFI', 'AGFI', 'NFI', 'TLI']):
                        if val > 0.90:
                            colors.append(C_TEAL)
                        else:
                            colors.append(C_CORAL)
                        clean_values.append(val)
                    elif any(x in str(measure) for x in ['RMSEA', 'SRMR']):
                        if val < 0.10:
                            colors.append(C_TEAL)
                        else:
                            colors.append(C_CORAL)
                        clean_values.append(val)
                    elif 'p-value' in str(measure).lower():
                        if val > 0.05:
                            colors.append(C_TEAL) # Non-significant p-value means good fit for Chi-square
                        else:
                            colors.append(C_CORAL)
                        clean_values.append(val)
                    else:
                        colors.append(C_GRAY)
                        clean_values.append(val)
                except:
                    colors.append(C_GRAY)
                    clean_values.append(0.0)

        bars = plt.barh(indices, clean_values, color=colors, edgecolor='white', linewidth=0.6, height=0.55)

        for bar, val in zip(bars, values):
            try:
                val_float = float(val)
                if np.isfinite(val_float):
                    plt.text(val_float + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{val_float:.3f}', va='center', fontsize=9)
                else:
                    plt.text(0.01, bar.get_y() + bar.get_height()/2,
                            'N/A', va='center', fontsize=9)
            except:
                plt.text(0.01, bar.get_y() + bar.get_height()/2,
                        str(val), va='center', fontsize=9)

        plt.xlabel('Index Value')
        plt.title(f'Model Fit Indices (Estimator: {estimator})', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.3)

        plt.figtext(0.01, 0.01,
                   f"Cutoffs: >0.90 (CFI, GFI, AGFI, NFI, TLI), <0.10 (RMSEA, SRMR) | Green = meets criterion, Red = fails, Grey = Comparative metric",
                   ha='left', fontsize=9)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Fit indices visualization saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Error visualizing fit indices: {str(e)}")

if fit_indices is not None:
    try:
        visualize_fit_indices(fit_indices, output_path=f'{output_dir}/fit_indices.png')
    except Exception as e:
        print(f"Error visualizing fit indices: {str(e)}")

def visualize_r_squared(r_squared_df=None, output_path=None):
    """Create a visualization of R-squared values for endogenous variables"""
    if r_squared_df is None:
        print("Cannot visualize R-squared: missing data")
        return

    r_squared_df = r_squared_df.copy()
    if 'R_squared' in r_squared_df.columns:
        r_squared_df = r_squared_df[np.isfinite(r_squared_df['R_squared'])]
        if r_squared_df.empty:
            print("No valid R-squared values to visualize")
            return

    plt.figure(figsize=(10, 6))

    r_squared_df = r_squared_df.sort_values(by='R_squared')

    ramp = [C_AMBER, C_CORAL, C_BLUE, C_TEAL, C_PURPLE, C_GREEN]
    bars = plt.barh(
        r_squared_df['Variable'], r_squared_df['R_squared'],
        color=ramp[:len(r_squared_df)],
        edgecolor='white', linewidth=0.6, height=0.55
    )

    # Add value labels with safety check
    for bar in bars:
        width = bar.get_width()
        if np.isfinite(width):
            plt.text(
                width + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va='center'
            )

    plt.xlabel('R-squared (Variance Explained)')
    plt.title('Variance Explained by the Model')
    plt.xlim(0, max(1.0, r_squared_df['R_squared'].max() + 0.1))
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"R-squared visualization saved to {output_path}")
    plt.close()

if results is not None:
    try:
        visualize_path_coefficients(results, output_path=f'{output_dir}/path_coefficients.png')
    except Exception as e:
        print(f"Error visualizing path coefficients: {str(e)}")

visualize_mediation_effects(output_path=f'{output_dir}/mediation_effects.png')

if r_squared is not None:
    try:
        visualize_r_squared(r_squared, output_path=f'{output_dir}/r_squared_values.png')
    except Exception as e:
        print(f"Error visualizing R-squared values: {str(e)}")

q11_attributes_sanitized_lower = [attr.replace(' ', '_').lower() for attr in q11_attributes]
motivation_cols = [f'Q11_{attr}' for attr in q11_attributes_sanitized_lower]

plt.figure(figsize=(16, 10))
plt.suptitle('Key Relationships in SEM Model', fontsize=16)

gs = GridSpec(2, 3, figure=plt.gcf())

ax1 = plt.subplot(gs[0, 0])
sns.regplot(x='Q4_numeric', y='Q9_numeric', data=data_numeric, ax=ax1,
            line_kws={"color": C_CORAL, "linewidth": 1.8},
            scatter_kws={"alpha": 0.55, "s": 30, "color": C_BLUE, "edgecolors": "white", "linewidths": 0.5})
ax1.set_title('Social Media Usage vs\nDigital Fashion Awareness')
ax1.set_xlabel('Social Media Usage (Hours)')
ax1.set_ylabel('Digital Fashion Awareness')

ax2 = plt.subplot(gs[0, 1])
sns.regplot(x='Q9_numeric', y='Motivation_Score', data=data_numeric, ax=ax2,
            line_kws={"color": C_CORAL, "linewidth": 1.8},
            scatter_kws={"alpha": 0.55, "s": 30, "color": C_BLUE, "edgecolors": "white", "linewidths": 0.5})
ax2.set_title('Digital Fashion Awareness vs\nMotivation Score')
ax2.set_xlabel('Digital Fashion Awareness')
ax2.set_ylabel('Motivation Score')

ax3 = plt.subplot(gs[0, 2])
sns.regplot(x='Motivation_Score', y='Q12_numeric', data=data_numeric, ax=ax3,
            line_kws={"color": C_CORAL, "linewidth": 1.8},
            scatter_kws={"alpha": 0.55, "s": 30, "color": C_BLUE, "edgecolors": "white", "linewidths": 0.5})
ax3.set_title('Motivation Score vs\nWillingness to Pay')
ax3.set_xlabel('Motivation Score')
ax3.set_ylabel('Willingness to Pay')

ax4 = plt.subplot(gs[1, 0])
sns.regplot(x='Q14', y='Q16', data=data_numeric, ax=ax4,
            line_kws={"color": C_CORAL, "linewidth": 1.8},
            scatter_kws={"alpha": 0.55, "s": 30, "color": C_BLUE, "edgecolors": "white", "linewidths": 0.5})
ax4.set_title('Environmental Awareness vs\nWillingness to Invest')
ax4.set_xlabel('Environmental Awareness')
ax4.set_ylabel('Willingness to Invest')

ax5 = plt.subplot(gs[1, 1])
sns.regplot(x='Q9_numeric', y='Q16', data=data_numeric, ax=ax5,
            line_kws={"color": C_CORAL, "linewidth": 1.8},
            scatter_kws={"alpha": 0.55, "s": 30, "color": C_BLUE, "edgecolors": "white", "linewidths": 0.5})
ax5.set_title('Digital Fashion Awareness vs\nWillingness to Invest')
ax5.set_xlabel('Digital Fashion Awareness')
ax5.set_ylabel('Willingness to Invest')

ax6 = plt.subplot(gs[1, 2])
sns.regplot(x='Q12_numeric', y='Q16', data=data_numeric, ax=ax6,
            line_kws={"color": C_CORAL, "linewidth": 1.8},
            scatter_kws={"alpha": 0.55, "s": 30, "color": C_BLUE, "edgecolors": "white", "linewidths": 0.5})
ax6.set_title('Willingness to Pay vs\nInvestment Likelihood')
ax6.set_xlabel('Willingness to Pay')
ax6.set_ylabel('Investment Likelihood')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{output_dir}/key_relationships.png', dpi=300, bbox_inches='tight')
print(f"Key relationships visualization saved to {output_dir}/key_relationships.png")
plt.close()

# Using Spearman's ρ for ordinal outcomes (Q12_numeric, Q16) and binary predictors (Q11_*)
motivation_corr = pd.DataFrame()
target_vars = ['Q12_numeric', 'Q16']
target_labels = ['Willingness to Pay', 'Investment Likelihood']

# Use the correct column names when calculating correlations with Spearman's ρ
for var, label in zip(target_vars, target_labels):
    corrs = []
    for attr, attr_clean in zip(q11_attributes, q11_attributes_sanitized_lower):
        col_name = f'Q11_{attr_clean}'
        corrs.append(data_numeric[col_name].corr(data_numeric[var], method='spearman'))
    motivation_corr[label] = corrs

motivation_corr.index = q11_attributes

plt.figure(figsize=(10, 8))
sns.heatmap(motivation_corr, annot=True,
    cmap=LinearSegmentedColormap.from_list('motiv', ['#D85A30','#F1EFE8','#1D9E75']),
    fmt='.2f', center=0)
plt.title('Motivation Factors: Correlation with Key Outcomes', fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/motivation_correlations.png', dpi=300)
print(f"Motivation correlations saved to {output_dir}/motivation_correlations.png")
plt.close()

# Using Spearman's ρ for consistency with ordinal/binary variables
key_vars = ['Q4_numeric', 'Q9_numeric', 'Q12_numeric', 'Q14', 'Q16', 'Motivation_Score']
key_labels = ['Social Media Usage', 'Digital Fashion Familiarity',
              'Willingness to Pay', 'Environmental Awareness',
              'Investment Likelihood', 'Motivation Score']

overall_corr = data_numeric[key_vars].corr(method='spearman')

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(overall_corr, dtype=bool), k=1)
ax = sns.heatmap(overall_corr, mask=mask, annot=True,
    cmap=LinearSegmentedColormap.from_list('overall', ['#D85A30','#F1EFE8','#7F77DD']),
    fmt='.2f', xticklabels=key_labels, yticklabels=key_labels)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')
plt.title('Overall Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/overall_correlations.png', dpi=300)
print(f"Overall correlation matrix saved to {output_dir}/overall_correlations.png")
plt.close()

if 'Age' in data.columns and 'Gender' in data.columns:
    try:
        plt.figure(figsize=(14, 8))
        plt.suptitle('Demographic Analysis of Key Outcomes', fontsize=16)

        gs = GridSpec(1, 2, figure=plt.gcf())

        ax1 = plt.subplot(gs[0, 0])
        if len(data['Age'].unique()) <= 10:  # Only if we have reasonable number of age groups
            sns.boxplot(x='Age', y='Q12_numeric', hue='Gender', data=data_numeric, ax=ax1)
            ax1.set_title('Willingness to Pay by Demographics')
            ax1.set_xlabel('Age Group')
            ax1.set_ylabel('Willingness to Pay')
            plt.setp(ax1.get_xticklabels(), rotation=45)

        ax2 = plt.subplot(gs[0, 1])
        if len(data['Age'].unique()) <= 10:  # Only if we have reasonable number of age groups
            sns.boxplot(x='Age', y='Q16', hue='Gender', data=data_numeric, ax=ax2)
            ax2.set_title('Investment Likelihood by Demographics')
            ax2.set_xlabel('Age Group')
            ax2.set_ylabel('Investment Likelihood')
            plt.setp(ax2.get_xticklabels(), rotation=45)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{output_dir}/demographic_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Demographic analysis visualization saved to {output_dir}/demographic_analysis.png")
        plt.close()
    except Exception as e:
        print(f"Error generating demographic analysis: {str(e)}")
else:
    print("Note: Demographic analysis skipped (Age and Gender columns not found in data)")

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    cluster_vars = ['Q4_numeric', 'Q9_numeric', 'Q14', 'Motivation_Score']

    cluster_data = data_numeric[cluster_vars].dropna()

    if len(cluster_data) > 10:  # Only proceed if we have sufficient data
        X = StandardScaler().fit_transform(cluster_data)

        k = min(4, len(cluster_data) // 20 + 2)  # Simple heuristic, at least 2, max 4 clusters

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)

        cluster_data_labeled = cluster_data.copy()
        cluster_data_labeled['Cluster'] = kmeans.labels_

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Q9_numeric', y='Motivation_Score', hue='Cluster',
                        data=cluster_data_labeled, palette='viridis', s=80)
        plt.title('Customer Segments by Digital Fashion Awareness and Motivation', fontsize=14)
        plt.xlabel('Digital Fashion Awareness', fontsize=12)
        plt.ylabel('Motivation Score', fontsize=12)
        
        plt.xticks([1, 2, 3], ['Unaware (1)', 'Uncertain (2)', 'Aware (3)'])

        centers = kmeans.cluster_centers_

        # Convert centers to DataFrame with explicit column names for safe selection
        centers_df = pd.DataFrame(centers, columns=cluster_vars)

        scaler_for_centers = StandardScaler().fit(cluster_data[['Q9_numeric', 'Motivation_Score']])

        # Safely inverse transform using explicit column names
        center_points = scaler_for_centers.inverse_transform(centers_df[['Q9_numeric', 'Motivation_Score']])

        plt.scatter(center_points[:, 0], center_points[:, 1],
                    s=200, marker='X', c='red', edgecolor='black')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/customer_segments.png', dpi=300)
        print(f"Customer segmentation visualization saved to {output_dir}/customer_segments.png")
        plt.close()
except Exception as e:
    print(f"Skipping segmentation analysis: {str(e)}")

plt.figure(figsize=(16, 10))
plt.suptitle('Distributions of Key Variables', fontsize=16)

gs = GridSpec(2, 3, figure=plt.gcf())

dist_vars = [
    {'var': 'Q4_numeric', 'label': 'Social Media Usage', 'pos': (0, 0)},
    {'var': 'Q9_numeric', 'label': 'Digital Fashion Awareness', 'pos': (0, 1)},
    {'var': 'Motivation_Score', 'label': 'Motivation Score', 'pos': (0, 2)},
    {'var': 'Q12_numeric', 'label': 'Willingness to Pay', 'pos': (1, 0)},
    {'var': 'Q14', 'label': 'Environmental Awareness', 'pos': (1, 1)},
    {'var': 'Q16', 'label': 'Investment Likelihood', 'pos': (1, 2)}
]

for item in dist_vars:
    ax = plt.subplot(gs[item['pos']])
    sns.histplot(data_numeric[item['var']], kde=True, ax=ax)
    ax.set_title(f'Distribution: {item["label"]}')
    ax.set_xlabel(item["label"])
    ax.set_ylabel('Count')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{output_dir}/variable_distributions.png', dpi=300, bbox_inches='tight')
print(f"Distribution analysis visualization saved to {output_dir}/variable_distributions.png")
plt.close()

try:
    data_numeric['Q14_binned'] = pd.qcut(
        data_numeric['Q14'],
        3,
        labels=['Low', 'Medium', 'High']
    )

    plt.figure(figsize=(12, 9))
    sns.lmplot(
        x='Q9_numeric',
        y='Q16',
        hue='Q14_binned',
        data=data_numeric.dropna(subset=['Q9_numeric', 'Q16', 'Q14_binned']),
        palette='viridis',
        height=8,
        aspect=1.2,
        scatter_kws={'alpha': 0.6, 's': 60},
        line_kws={'linewidth': 2}
    )

    plt.title(
        'Interaction Effect: Environmental Awareness Moderates\nDigital Fashion Awareness → Investment Likelihood',
        fontsize=14, y=1.05)
    plt.xlabel('Digital Fashion Awareness', fontsize=12)
    plt.ylabel('Investment Likelihood', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/interaction_effects.png', dpi=300, bbox_inches='tight')
    print(f"Interaction effects visualization saved to {output_dir}/interaction_effects.png")
    plt.close()
except Exception as e:
    print(f"Skipping interaction effects visualization: {str(e)}")

try:
    motivation_means = np.array([data_numeric[f'Q11_{attr}'].mean() for attr in q11_attributes_sanitized_lower])

    if np.all(np.isfinite(motivation_means)):
        angles = np.linspace(0, 2 * np.pi, len(q11_attributes), endpoint=False).tolist()
        motivation_means_list = motivation_means.tolist()

        angles += [angles[0]]
        motivation_means_list += [motivation_means_list[0]]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        ax.plot(angles, motivation_means_list, 'o-', linewidth=2.5, color=C_PURPLE, markersize=8)
        ax.fill(angles, motivation_means_list, alpha=0.25, color=C_PURPLE)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(q11_attributes, size=11, weight='bold')

        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)

        ax.grid(True, linestyle='--', alpha=0.7)

        ax.set_title('Motivation Factors Analysis\n(Mean Prevalence)', fontsize=15, pad=20, weight='bold')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/motivation_radar.png', dpi=300, bbox_inches='tight')
        print(f"✓ Motivation radar chart saved to {output_dir}/motivation_radar.png")
        plt.close()
    else:
        print("Skipping radar chart: non-finite values in motivation data")

except Exception as e:
    print(f"Skipping radar chart visualization: {str(e)}")
    import traceback
    traceback.print_exc()

try:
    try:
        params_df = pd.read_csv('sem_results/parameter_estimates.csv')
        param_dict = {}
        for _, row in params_df[params_df['op'] == '~'].iterrows():
            param_dict[(row['rval'], row['lval'])] = (row['Estimate'], row['p-value'])
    except:
        param_dict = {}
        print("Note: parameter_estimates.csv not found or missing columns, path coefficients will use fallback values.")

    def create_network_diagram(edges, positions, title, output_filename, node_labels, p_dict=None, corr_df=None):
        fig, ax = plt.subplots(figsize=(16, 9))
        
        x_coords = [pos[0] for pos in positions.values()]
        y_coords = [pos[1] for pos in positions.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        ax.set_xlim(x_min - 3.0, x_max + 3.0)
        ax.set_ylim(y_min - 4.5, y_max + 2.5)  # Extended bottom padding significantly for legend
        
        for edge in edges:
            u = edge[0]
            v = edge[1]
            rad = edge[2] if len(edge) > 2 else 0.0
            t_val = edge[3] if len(edge) > 3 else 0.5
            offset = edge[4] if len(edge) > 4 else (0, 0.3)

            pos_u = positions[u]
            pos_v = positions[v]
            
            edge_color = '#888780' # Default gray
            edge_label = ""

            if p_dict and (u, v) in p_dict:
                est, pval = p_dict[(u, v)]
                edge_color = '#1D9E75' if est > 0 else '#D85A30' # Teal positive, Coral negative
                stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                edge_label = f"β = {est:.2f}{stars}"
            elif corr_df is not None and u in corr_df.columns and v in corr_df.columns:
                corr = corr_df.loc[u, v]
                edge_color = '#1D9E75' if corr > 0 else '#D85A30'
                edge_label = f"r = {corr:.2f}"

            ax.annotate("",
                        xy=pos_v, xycoords='data',
                        xytext=pos_u, textcoords='data',
                        arrowprops=dict(arrowstyle="-|>", color=edge_color,
                                        shrinkA=55, shrinkB=55,
                                        connectionstyle=f"arc3,rad={rad}", lw=2.5, alpha=0.85))

            if edge_label:
                base_x = (1 - t_val) * pos_u[0] + t_val * pos_v[0]
                base_y = (1 - t_val) * pos_u[1] + t_val * pos_v[1]

                # Apply absolute offset to space text safely away from the arrow string
                lx = base_x + offset[0]
                ly = base_y + offset[1]

                ax.text(lx, ly, edge_label, ha='center', va='center',
                        fontsize=10, fontweight='bold', color=edge_color,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor='white', edgecolor='none', alpha=0.9))

        for node, pos in positions.items():
            label = node_labels.get(node, node)
            ax.text(pos[0], pos[1], label,
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.6", facecolor='#2B3A42', 
                              edgecolor='#D3D3D3', linewidth=1.5, alpha=0.95, zorder=5))
                    
        legend_text = (
            "Relationship Information:\n"
            "β = Standardized Path Coefficient (SEM structural path)\n"
            "r = Pearson Correlation (Composite Formation path)\n"
            "■ Green Arrow = Positive Relationship  |  ■ Orange Arrow = Negative Relationship\n"
            "Significance levels: *** p<0.001,  ** p<0.01,  * p<0.05,  ns: not significant"
        )
        ax.text(x_min - 1.5, y_min - 1.5, legend_text, ha='left', va='top',
                fontsize=11, color='#333333', 
                bbox=dict(boxstyle="square,pad=0.8", facecolor='#FAFAF9', alpha=0.95, edgecolor='#C0C0C0', zorder=10))
        
        plt.title(title, fontsize=18, pad=20, fontweight='bold', color='#333333')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{output_filename}', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Network diagram saved to {output_dir}/{output_filename}")

    node_labels = {
        'Q4': 'Social Media\nUsage (Q4)',
        'Q9': 'Digital Fashion\nFamiliarity (Q9)',
        'motivation_count': 'Purchase\nMotivation Index',
        'Q12': 'Willingness\nTo Pay (Q12)',
        'Q14': 'Environmental\nConsciousness (Q14)',
        'Q16': 'Investment\nLikelihood (Q16)'
    }

    # Use Spearman's ρ for network diagram edges - more appropriate for ordinal/binary variables
    corr_matrix_val = data_numeric.select_dtypes(include=[np.number]).corr(method='spearman')

    full_edges = [
        ('Q4', 'Q9', 0.0, 0.45, (0, 0.4)),
        ('Q9', 'Q12', 0.0, 0.4, (-0.2, 0.5)),
        ('Q14', 'Q12', 0.0, 0.4, (-0.2, -0.5)),
        ('motivation_count', 'Q12', 0.0, 0.6, (0.0, -0.6)),
        ('Q12', 'Q16', 0.0, 0.5, (0, 0.4)),
        ('Q9', 'Q16', -0.15, 0.35, (-0.2, 1.4)),  # Arching over Q12
        ('motivation_count', 'Q16', 0.15, 0.35, (-0.2, -1.4))  # Arching under Q12
    ]
    full_pos = {
        'Q4': (0, 6),
        'Q14': (0, 0),
        'Q9': (5, 6),
        'motivation_count': (5, 0),
        'Q12': (10, 3),
        'Q16': (16, 3)
    }
    create_network_diagram(full_edges, full_pos, "Full SEM Structural Model with Path Values",
                           "network_full_model.png", node_labels, p_dict=param_dict, corr_df=corr_matrix_val)

    p1_edges = [
        ('Q4', 'Q9', 0.0, 0.5, (0, 0.4)),
        ('Q9', 'Q12', 0.0, 0.45, (-0.2, 0.5)),
        ('motivation_count', 'Q12', 0.0, 0.55, (-0.2, -0.5))
    ]
    p1_pos = {
        'Q4': (0, 4),
        'Q9': (6, 4),
        'motivation_count': (6, 0),
        'Q12': (12, 2)
    }
    create_network_diagram(p1_edges, p1_pos, "Pathway 1:\nSocial Media → Familiarity → Willingness to Pay",
                           "network_pathway_1.png", node_labels, p_dict=param_dict, corr_df=corr_matrix_val)

    p2_edges = [
        ('Q14', 'Q12', 0.0, 0.4, (0, 0.4)),
        ('Q12', 'Q16', 0.0, 0.5, (0, 0.4)),
        ('Q9', 'Q16', -0.15, 0.65, (0, 1.2)),
        ('motivation_count', 'Q16', 0.15, 0.65, (0, -1.2))
    ]
    p2_pos = {
        'Q14': (0, 0),
        'Q9': (4, 5),
        'motivation_count': (4, -5),
        'Q12': (6, 0),
        'Q16': (13, 0)
    }
    create_network_diagram(p2_edges, p2_pos, "Pathway 2:\nEnvironmental Consciousness → Willingness to Pay → Investment",
                           "network_pathway_2.png", node_labels, p_dict=param_dict, corr_df=corr_matrix_val)

    p3_node_labels = node_labels.copy()
    p3_pos = {
        'motivation_count': (14, 5),
        'Q12': (20, 8),
        'Q16': (20, 2)
    }
    p3_edges = [
        ('motivation_count', 'Q12', 0.0, 0.5, (-0.5, 0.5)),
        ('motivation_count', 'Q16', 0.0, 0.5, (-0.5, -0.5)),
        ('Q12', 'Q16', 0.0, 0.5, (0.5, 0))
    ]
    
    for i, attr in enumerate(q11_attributes_sanitized_lower):
        node_name = f'Q11_{attr}'
        p3_node_labels[node_name] = q11_attributes[i]
        y_pos = 10.0 - (i * 1.25)
        p3_pos[node_name] = (0, y_pos)
        p3_edges.append((node_name, 'motivation_count', 0.0, 0.28, (0, 0.35)))
        
    create_network_diagram(p3_edges, p3_pos, "Pathway 3:\nMotivation Component Formation and Impact", 
                           "network_pathway_3.png", p3_node_labels, p_dict=param_dict, corr_df=corr_matrix_val)

except Exception as e:
    print(f"Error generating network diagrams: {str(e)}")
    import traceback
    traceback.print_exc()

visualization_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
print(f"All visualization outputs saved to: {output_dir}")

try:
    if os.path.exists('sem_results/qq_plots.png'):
        print("\n✓ Q-Q plots for normality assessment saved to: sem_results/qq_plots.png")
        print("  Review these plots to evaluate multivariate normality assumption")
except Exception as e:
    print(f"Note: Q-Q plots may have been generated during SEM analysis: {str(e)}")

