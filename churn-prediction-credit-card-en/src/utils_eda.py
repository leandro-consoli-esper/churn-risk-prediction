# This notebook contains customized utility plotting functions used to facilitate the exploratory data analysis.

# Imports
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns

# Warnings
from warnings import filterwarnings
filterwarnings('ignore')

# Palette Setting
color_palette = ['#5cd1c5', '#f25c87', '#b8b5b4', '#007f66', '#063366', '#eee8e4', '#850885']
sns.set_palette(sns.color_palette(color_palette))

# Target Distribution Pie Chart
# =========================================================
def plot_pie(data, color_palette, labels):
    """
    Plot the churn target distribution as a pie chart.

    Parameters
    ----------
    data : pd.Series
        Count of each churn class.
    color_palette : list
        List of colors to apply to the chart.
    labels : list
        Labels for each churn class.
    """
    plt.figure(figsize=(4, 3))
    plt.pie(
        data.values,
        labels=labels,
        autopct='%1.1f%%',
        textprops={'fontsize': 10},
        startangle=90,
        colors=color_palette
    )
    plt.title('Target Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Bar Plots for Categorical Variables
# =========================================================
def plot_bars(data, cat_names, color_palette):
    """
    Create bar plots for each categorical variable.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with categorical columns.
    cat_names : list
        Titles to use for each subplot.
    color_palette : list
        Colors for the bar categories.
    """
    num_cols = len(data.columns)
    n_cols = 3
    n_rows = (num_cols // n_cols) + (num_cols % n_cols > 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(data.columns):
        counts = data[col].value_counts()
        counts.plot(
            kind='bar',
            ax=axes[i],
            color=color_palette,
            edgecolor="black"
        )
        axes[i].set_title(cat_names[i])
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', rotation=45)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Stacked Churn Proportion by Category
# =========================================================
def plot_cat_churn(data, columns, cat_names, color_palette):
    """
    Plot stacked bar charts of churn proportions for categorical features.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset including categorical columns and churn_flag.
    columns : list
        List of columns to plot.
    cat_names : list
        Friendly titles for each categorical column.
    color_palette : list
        Colors for active vs churn.
    """
    fig, axes = plt.subplots(1, len(columns), figsize=(18, 5))

    for i, col in enumerate(columns):
        prop = (
            data.groupby([col, 'churn_flag']).size()
            .groupby(level=0).apply(lambda x: x / x.sum())
            .unstack()
        )

        prop.index = prop.index.get_level_values(0).astype(str)

        prop.plot(
            kind='bar',
            stacked=True,
            ax=axes[i],
            color=color_palette,
            edgecolor='black'
        )

        axes[i].set_title(f'Churn Dist. for {cat_names[i]}', fontsize=13, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Proportion')
        axes[i].legend(['Active', 'Churn'])
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


# Pie Plots for Churn Rate by Categorical Variables
# =========================================================
def plot_pie_churn(data, cols_to_plot, color_palette):
    """
    Create side-by-side pie charts showing churn rate distribution
    for each categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing categorical columns and 'churn_flag'.
    cols_to_plot : list
        List of categorical columns to visualize.
    color_palette : list
        Colors to use in the pie charts.
    """
    n_cols = len(cols_to_plot)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    
    # Guarantee axes is iterable even when n_cols == 1
    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(cols_to_plot):
        churn_rate = (
            data.groupby(col)['churn_flag']
            .mean()
            .mul(100)
            .reset_index(name='churn_rate (%)')
            .sort_values(by=col)
        )

        axes[i].pie(
            churn_rate['churn_rate (%)'],
            labels=churn_rate[col],
            autopct='%1.1f%%',
            startangle=90,
            colors=color_palette
        )

        axes[i].set_title(f'Churn Rate by {col}', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show()


# Observed vs Expected Heatmaps (Chi-square)
# =========================================================
def plot_heatmap(contingency_table, data):
    """
    Plot two heatmaps: observed frequencies and expected frequencies.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        Observed contingency table for gender × churn.
    expected_df : pd.DataFrame
        Expected values under the null hypothesis.
    """
    # Update axis labels for readability
    contingency_table.index = ['Female', 'Male']
    contingency_table.columns = ['Active', 'Churn']

    data.index = ['Female', 'Male']
    data.columns = ['Active', 'Churn']

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    sns.heatmap(
        contingency_table, annot=True, fmt='d', cmap='Spectral',
        cbar=False, ax=axes[0]
    )
    axes[0].set_title('Observed Frequencies', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')

    sns.heatmap(
        data, annot=True, fmt='.1f', cmap='Spectral',
        cbar=False, ax=axes[1]
    )
    axes[1].set_title('Expected Frequencies (H₀)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')

    fig.suptitle(
        'Gender × Churn — Observed vs Expected Frequencies',
        fontsize=13, fontweight='bold', y=1.05
    )
    plt.tight_layout()
    plt.show()


# Histogram Distribution for Numerical Variables
# =========================================================
def plot_hist(data, num_var, color_palette):
    """
    Plot histograms for all numerical variables with churn overlay.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset including churn_flag.
    num_var : pd.DataFrame
        DataFrame containing only numerical variables.
    color_palette : list
        Colors for active vs churn.
    """
    n_cols = 3
    num_cols = len(num_var.columns)
    n_rows = (num_cols // n_cols) + (num_cols % n_cols > 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_var.columns):
        sns.histplot(
            data=data,
            x=col,
            hue="churn_flag",
            bins=25,
            palette=color_palette,
            kde=True,
            alpha=0.6,
            ax=axes[i]
        )
        axes[i].set_title(col, fontsize=18, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Violin Plots with Means for Numerical Variables
# =========================================================
def plot_violin(data, num_var, color_palette):
    """
    Plot violin charts for numerical variables with overlayed means.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset including churn_flag.
    num_var : pd.DataFrame
        Numerical variables only.
    color_palette : list
        Churn color palette.
    """
    fig, axes = plt.subplots(4, 4, figsize=(18, 15))
    axes = axes.flatten()

    for i, col in enumerate(num_var.columns):
        sns.violinplot(
            data=data,
            x="churn_flag",
            y=col,
            ax=axes[i],
            palette=color_palette,
            inner="quartile"
        )

        means = data.groupby("churn_flag")[col].mean()
        for j, mean in enumerate(means):
            axes[i].scatter(
                j, mean,
                color='#063366',
                s=60,
                marker="o",
                zorder=3,
                label="Mean" if i == 0 and j == 0 else ""
            )

        axes[i].set_title(col, fontsize=18)
        axes[i].set_ylabel('')
        axes[i].legend(loc="upper right", fontsize=12)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Outlier Verification Function
#======================================================
def outlier_check(data, features):   
    outliercounts = {}
    outlier_index = {}
    total_outliers = 0
    
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        feature_outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
        outlier_index[feature] = feature_outliers.index.tolist()
        outlier_count = len(feature_outliers)
        outliercounts[feature] = outlier_count
        total_outliers += outlier_count
    
    print(f'There are {total_outliers} outliers in the dataset.')
    print()
    print(f'Number (percentage) of outliers per feature: ')
    print()
    for feature, count in outliercounts.items():
        print(f'{feature}: {count} ({round(count/len(data)*100, 2)})%')

    return outlier_index, outliercounts, total_outliers


# Engineered Features Histogram Comparison
# =========================================================
def plot_enghist(data, features):
    """
    Plot histograms for engineered features with churn comparison.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame including engineered features and churn_flag.
    features : list
        List of engineered feature column names.
    """
    n_cols = 3
    num_features = len(features)
    n_rows = (num_features // n_cols) + (num_features % n_cols > 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        means = data.groupby('churn_flag')[col].mean().to_dict()
        title = (
            f"{col} — churn vs non-churn\n"
            f"Mean (non-churn=0): {means.get(0, 0):.2f} | "
            f"Mean (churn=1): {means.get(1, 0):.2f}"
        )

        sns.histplot(
            data=data,
            x=col,
            hue='churn_flag',
            bins=30,
            multiple='layer',
            palette={0: '#5cd1c5', 1: '#f25c87'},
            alpha=0.5,
            ax=axes[i]
        )

        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Count')
        axes[i].legend(title='Churn Flag', labels=['Non-Churn', 'Churn'])

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Risk Level Pie Chart
# =========================================================
def plot_risk_pie(risk_churn, color_palette):
    """
    Plot churn rate by risk level as a pie chart.

    Parameters
    ----------
    risk_churn : pd.DataFrame
        Must contain columns ['risk_level', 'churn_flag'] aggregated.
    color_palette : list
        Colors for each risk level.
    """
    plt.figure(figsize=(4, 4))
    plt.pie(
        risk_churn['churn_flag'],
        labels=risk_churn['risk_level'],
        textprops={'fontsize': 8, 'color': 'black'},
        autopct='%1.1f%%',
        startangle=90,
        colors=color_palette,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )

    plt.title('Churn Rate by Risk Level (%)', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.show()