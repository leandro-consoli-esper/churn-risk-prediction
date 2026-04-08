# Este notebook contém funções de plotagem customizadas que serão usadas para facilitar o trabalho de análise exploratória de dados

# Importações de bibliotecas
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns

# Avisos
from warnings import filterwarnings
filterwarnings('ignore')

# Configuração de Paleta
color_palette = ['#5cd1c5', '#f25c87', '#b8b5b4', '#007f66', '#063366', '#eee8e4', '#850885']
sns.set_palette(sns.color_palette(color_palette))

# Gráfico de Pizza com a Distribuição da Variável Alvo
# =========================================================
def plot_pie(data, color_palette, labels):
    """
    Plota a distribuição da variável alvo de churn em formato de gráfico de pizza.

    Parâmetros
    ----------
    data : pd.Series
        Contagem de cada classe de churn.
    color_palette : list
        Lista de cores a serem aplicadas no gráfico.
    labels : list
        Rótulos para cada classe de churn.
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


# Gráficos de Barras para Variáveis Categóricas
# =========================================================
def plot_bars(data, cat_names, color_palette):
    """
    Cria gráficos de barras para cada variável categórica.

    Parâmetros
    ----------
    data : pd.DataFrame
        DataFrame contendo colunas categóricas.
    cat_names : list
        Títulos a serem utilizados para cada subplot.
    color_palette : list
        Cores para as categorias nos gráficos de barras.
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

    # Remove subplots vazios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Proporção de Churn Agrupada por Categoria
# =========================================================
def plot_cat_churn(data, columns, cat_names, color_palette):
    """
    Plota gráficos de barras empilhadas das proporções de churn
    para variáveis categóricas.

    Parâmetros
    ----------
    data : pd.DataFrame
        Dataset completo contendo colunas categóricas e a variável churn_flag.
    columns : list
        Lista de colunas a serem plotadas.
    cat_names : list
        Títulos amigáveis para cada variável categórica.
    color_palette : list
        Cores para clientes ativos vs churn.
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


# Gráficos de Pizza com a Taxa de Churn por Variável Categórica
# =========================================================
def plot_pie_churn(data, cols_to_plot, color_palette):
    """
    Cria gráficos de pizza lado a lado mostrando a distribuição
    da taxa de churn para cada variável categórica.

    Parâmetros
    ----------
    df : pd.DataFrame
        Dataset completo contendo colunas categóricas e 'churn_flag'.
    cols_to_plot : list
        Lista de colunas categóricas a serem visualizadas.
    color_palette : list
        Cores a serem utilizadas nos gráficos de pizza.
    """
    n_cols = len(cols_to_plot)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    
    # Garante que 'axes' seja iterável, mesmo quando n_cols == 1
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


# Mapa de Calor(Chi-square): Observado vs Esperado
# =========================================================
def plot_heatmap(contingency_table, data):
    """
    Plota dois mapas de calor: frequências observadas e frequências esperadas.

    Parâmetros
    ----------
    contingency_table : pd.DataFrame
        Tabela de contingência observada para gênero × churn.
    expected_df : pd.DataFrame
        Valores esperados sob a hipótese nula.
    """
    # Atualiza o nome dos eixos para legibilidade
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


# Histograma de Distribuição para Variáveis Numéricas
# =========================================================
def plot_hist(data, num_var, color_palette):
    """
    Plota histogramas para todas as variáveis numéricas com sobreposição de churn.

    Parâmetros
    ----------
    data : pd.DataFrame
        Dataset completo incluindo churn_flag.
    num_var : pd.DataFrame
        DataFrame contendo apenas variáveis numéricas.
    color_palette : list
        Cores para clientes ativos vs churn.
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


# Gráficos de Violino com Média para Variáveis Numéricas
# =========================================================
def plot_violin(data, num_var, color_palette):
    """
    Plota gráficos de violino para variáveis numéricas com médias sobrepostas.

    Parâmetros
    ----------
    data : pd.DataFrame
        Dataset incluindo churn_flag.
    num_var : pd.DataFrame
        Apenas variáveis numéricas.
    color_palette : list
        Paleta de cores para churn.
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


# Função para Verificação de Outliers
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


# Histograma de Comparação das Variáveis Combinadas
# =========================================================
def plot_enghist(data, features):
    """
    Plota histogramas para variáveis derivadas com comparação de churn.

    Parâmetros
    ----------
    data : pd.DataFrame
        DataFrame incluindo variáveis derivadas e churn_flag.
    features : list
        Lista com os nomes das variáveis derivadas.
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


# Gráfico de Pizza do Nível de Risco
# =========================================================
def plot_risk_pie(risk_churn, color_palette):
    """
    Plota a taxa de churn por nível de risco em formato de gráfico de pizza.

    Parâmetros
    ----------
    risk_churn : pd.DataFrame
        Deve conter colunas ['risk_level', 'churn_flag'] agregadas.
    color_palette : list
        Cores para cada nível de risco.
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