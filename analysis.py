import pandas as pd
import numpy as np
import pyreadstat
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import glob
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CONFIG = {
    "REF_GENO": "GG",
    "OUTPUT_MD": "Schizo_Association_Final_Report.md",
    "FIGURE_DIR": "figures",
}

P_VALUE_DISPLAY_THRESHOLD = 0.0001

os.makedirs(CONFIG["FIGURE_DIR"], exist_ok=True)


def print_status(msg):
    print(f">>> [执行中] {msg}", flush=True)


# ─────────────────────────────────────────────
# Outlier validation helpers
# ─────────────────────────────────────────────

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Return a boolean mask marking IQR-based outliers (True = outlier)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def validate_and_report_outliers(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    """
    Flag IQR outliers in *col*, print a summary, and return the dataframe
    with an extra boolean column ``{col}_OUTLIER``.
    """
    mask = detect_outliers_iqr(df[col].dropna())
    full_mask = pd.Series(False, index=df.index)
    full_mask.loc[mask.index] = mask
    df = df.copy()
    df[f"{col}_OUTLIER"] = full_mask
    n_out = full_mask.sum()
    print_status(
        f"异常值检测 [{label}]: 共 {len(df)} 例，IQR 法识别 {n_out} 个异常值"
        + (f" (占 {n_out/len(df)*100:.1f}%)" if len(df) > 0 else "")
    )
    if n_out:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        print(f"    正常范围: [{lo:.2f}, {hi:.2f}]  "
              f"异常值范围: {df.loc[full_mask, col].min():.2f}~{df.loc[full_mask, col].max():.2f}")
    return df


# ─────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(CONFIG["FIGURE_DIR"], name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_genotype_distribution(df: pd.DataFrame, geno_col: str = 'GENO_CLEAN') -> str:
    """Stacked bar chart: genotype frequencies for cases vs controls."""
    geno_order = ['AA', 'AG', 'GG']
    counts = (
        df.groupby(['Y', geno_col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=geno_order, fill_value=0)
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['#4e79a7', '#f28e2b', '#e15759']

    for ax, data, title in zip(axes, [counts, pct], ['Count', 'Frequency (%)']):
        bottom = np.zeros(len(data))
        for i, geno in enumerate(geno_order):
            vals = data[geno].values if geno in data.columns else np.zeros(len(data))
            ax.bar(data.index.map({0: 'Control', 1: 'Case'}),
                   vals, bottom=bottom, color=colors[i], label=geno, edgecolor='white')
            bottom += vals
        ax.set_title(f'Genotype Distribution ({title})')
        ax.set_xlabel('Group')
        ax.set_ylabel(title)
        ax.legend(title='Genotype')

    fig.suptitle('RS2071313 Genotype Distribution', fontsize=13, fontweight='bold')
    fig.tight_layout()
    return _save(fig, 'genotype_distribution.png')


def plot_age_boxplot(df: pd.DataFrame, age_col: str, p_value: float) -> str:
    """Box plot of age by group, with individual outliers highlighted."""
    fig, ax = plt.subplots(figsize=(6, 5))
    groups = {0: 'Control', 1: 'Case'}
    data_list = [df[df['Y'] == g][age_col].dropna().values for g in [0, 1]]
    outlier_flags = [df[df['Y'] == g].get(f'{age_col}_OUTLIER', pd.Series(dtype=bool)).values
                     for g in [0, 1]]

    bp = ax.boxplot(data_list, labels=list(groups.values()),
                    patch_artist=True, medianprops={'color': 'black', 'linewidth': 2})
    colors = ['#4e79a7', '#e15759']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay outlier points
    for i, (vals, flags) in enumerate(zip(data_list, outlier_flags), start=1):
        if len(flags) == len(vals):
            out_vals = vals[flags]
        else:
            out_vals = np.array([])
        if len(out_vals):
            ax.scatter([i] * len(out_vals), out_vals,
                       color='red', zorder=5, s=40, label='Outlier' if i == 1 else None)

    if p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    ax.set_title(f'Age Distribution by Group\n(Mann-Whitney U, P={p_value:.4f} {sig})')
    ax.set_ylabel('Age (years)')
    has_outliers = any(bool(flags.any()) for flags in outlier_flags if len(flags))
    if has_outliers:
        ax.legend(loc='upper right')
    fig.tight_layout()
    return _save(fig, 'age_boxplot.png')


def plot_hwe(obs: list, exp: list, hwe_p: float) -> str:
    """Grouped bar chart comparing observed vs expected genotype counts (HWE)."""
    geno_labels = ['AA', 'AG', 'GG']
    x = np.arange(len(geno_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, obs, width, label='Observed', color='#4e79a7')
    ax.bar(x + width / 2, exp, width, label='Expected (HWE)', color='#f28e2b', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(geno_labels)
    ax.set_title(f'Hardy-Weinberg Equilibrium (Control)\nχ² P = {hwe_p:.4f}')
    ax.set_ylabel('Count')
    ax.legend()
    fig.tight_layout()
    return _save(fig, 'hwe_test.png')


def plot_forest(res_df: pd.DataFrame) -> str:
    """Forest plot of OR with 95% CI for each genetic model."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(res_df) * 1.2)))
    y_pos = np.arange(len(res_df))[::-1]

    for i, (_, row) in enumerate(res_df.iterrows()):
        lo, hi = row['CI_LO'], row['CI_HI']
        color = '#e15759' if row.get('P', 1) < 0.05 else '#4e79a7'
        yi = y_pos[i]
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2)
        ax.scatter(row['OR'], yi, color=color, zorder=5, s=60)
        if row['P'] >= P_VALUE_DISPLAY_THRESHOLD:
            p_label = f"P={row['P']:.4f}"
        else:
            p_label = f"P<{P_VALUE_DISPLAY_THRESHOLD}"
        ax.text(hi + 0.02, yi, f"OR={row['OR']:.2f}  {p_label}", va='center', fontsize=8)

    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(res_df['Model'].tolist())
    ax.set_xlabel('Odds Ratio (95% CI)')
    ax.set_title('Logistic Regression: Genetic Model Forest Plot')
    sig_patch = mpatches.Patch(color='#e15759', label='P < 0.05')
    ns_patch = mpatches.Patch(color='#4e79a7', label='P ≥ 0.05')
    ax.legend(handles=[sig_patch, ns_patch], loc='lower right')
    fig.tight_layout()
    return _save(fig, 'forest_plot.png')


def super_clean_id(x):
    if pd.isna(x): return ""
    nums = re.findall(r'\d+', str(x))
    return "".join(nums) if nums else str(x).strip()

try:
    print_status("启动增强版分析流程...")

    # --- 1. 定位并读取 Excel ---
    all_xlsx = [f for f in glob.glob("*.xlsx") if "~$" not in f]
    target_files = [f for f in all_xlsx if "RS2071313" in f.upper()]
    target_file = target_files[0] if target_files else all_xlsx[0]
    print_status(f"锁定基因型文件：{target_file}")
    
    df_geno = pd.read_excel(target_file, engine='openpyxl')
    df_geno.columns = [c.upper().strip() for c in df_geno.columns]

    # --- 2. 读取 SAV 并自动识别列名 ---
    sav_file = next(f for f in os.listdir('.') if f.endswith('.sav'))
    df_info, meta = pyreadstat.read_sav(sav_file, encoding='gbk')
    df_info.columns = [c.upper().strip() for c in df_info.columns]
    
    print_status(f"SAV 包含列名: {list(df_info.columns)}")

    # 自动映射关键列（兼容中文和常见缩写）
    col_map = {
        'ID': next((c for c in df_info.columns if any(k in c for k in ['SAMPLE', 'ID', '编号', '样本'])), 'SAMPLE'),
        'GROUP': next((c for c in df_info.columns if any(k in c for k in ['GROUP', '分组', 'DISEASE', 'CASE'])), 'GROUP'),
        'AGE': next((c for c in df_info.columns if any(k in c for k in ['AGE', '年龄', 'NL'])), 'AGE'),
        'GENDER': next((c for c in df_info.columns if any(k in c for k in ['GENDER', 'SEX', '性别', 'XB'])), 'GENDER')
    }
    print_status(f"列名映射结果: {col_map}")

    # --- 3. ID 匹配与平行样质控 ---
    df_info['ID_KEY'] = df_info[col_map['ID']].apply(super_clean_id)
    df_geno['ID_KEY'] = df_geno[df_geno.columns[0]].apply(super_clean_id)
    
    geno_col = next((c for c in df_geno.columns if any(k in c for k in ['分型', '基因型', 'GENO'])), df_geno.columns[-1])

    # 平行样一致性检查
    consistency = df_geno.groupby('ID_KEY')[geno_col].nunique()
    valid_ids = consistency[consistency == 1].index
    df_geno_clean = df_geno[df_geno['ID_KEY'].isin(valid_ids)].drop_duplicates('ID_KEY')
    
    # --- 4. 合并与清洗 ---
    df_merged = pd.merge(df_info, df_geno_clean[['ID_KEY', geno_col]], on='ID_KEY', how='inner')
    print_status(f"初步 ID 匹配成功数: {len(df_merged)}")

    def normalize_geno(g):
        if pd.isna(g): return None
        g = re.sub(r'[^A-Z]', '', str(g).upper())
        return g if g in ['AA', 'AG', 'GG'] else None

    df_merged['GENO_CLEAN'] = df_merged[geno_col].apply(normalize_geno)
    
    # 强制转换类型确保 dropna 正常
    for c in [col_map['GROUP'], col_map['AGE'], col_map['GENDER'], 'GENO_CLEAN']:
        df_merged[c] = df_merged[c].replace('', np.nan)

    df_final = df_merged.dropna(subset=[col_map['GROUP'], col_map['AGE'], col_map['GENDER'], 'GENO_CLEAN']).copy()
    
    # 统计缺失情况
    if df_final.empty:
        print_status("!!! 严重错误：清洗后数据为空。")
        print(f"--- 缺失值诊断 ---")
        print(df_merged[[col_map['GROUP'], col_map['AGE'], col_map['GENDER'], 'GENO_CLEAN']].isna().sum())
        exit()

    unique_groups = sorted(df_final[col_map['GROUP']].unique())
    df_final['Y'] = (df_final[col_map['GROUP']] == unique_groups[-1]).astype(int)
    print_status(f"质控完成：有效样本 {len(df_final)} 例。")

    # --- 4b. 异常值验证 ---
    print_status("开始异常值验证...")
    df_final = validate_and_report_outliers(df_final, col_map['AGE'], '年龄')
    age_outlier_col = f"{col_map['AGE']}_OUTLIER"
    n_age_outliers = int(df_final[age_outlier_col].sum())

    # --- 5. 统计分析 (逻辑保持不变) ---
    sex_tab = pd.crosstab(df_final['Y'], df_final[col_map['GENDER']])
    chi2_sex, p_sex, _, _ = stats.chi2_contingency(sex_tab)
    
    age_ctrl = df_final[df_final['Y'] == 0][col_map['AGE']]
    age_case = df_final[df_final['Y'] == 1][col_map['AGE']]
    u_stat, p_age = stats.mannwhitneyu(age_ctrl, age_case)
    
    # HWE
    ctrl_data = df_final[df_final['Y'] == 0]
    obs = [ctrl_data['GENO_CLEAN'].value_counts().get(g, 0) for g in ['AA', 'AG', 'GG']]
    p_all = (2 * obs[0] + obs[1]) / (2 * len(ctrl_data))
    exp = [(p_all**2)*len(ctrl_data), (2*p_all*(1-p_all))*len(ctrl_data), ((1-p_all)**2)*len(ctrl_data)]
    hwe_chi, hwe_p = stats.chisquare(obs, f_exp=exp)

    # Logistic
    df_final['G_CAT'] = pd.Categorical(df_final['GENO_CLEAN'], categories=[CONFIG["REF_GENO"], 'AG', 'AA'])
    models_def = {
        '显性模型 (AA+AG vs GG)': (df_final['GENO_CLEAN'] != CONFIG["REF_GENO"]).astype(int),
        '隐性模型 (AA vs AG+GG)': (df_final['GENO_CLEAN'] == 'AA').astype(int),
        '加性模型': df_final['G_CAT'].cat.codes,
        '纯合子模型 (AA vs GG)': df_final['GENO_CLEAN'].map({CONFIG["REF_GENO"]: 0, 'AA': 1})
    }

    results = []
    formula = f"Y ~ X + {col_map['AGE']} + {col_map['GENDER']}"
    for name, x_data in models_def.items():
        tdf = df_final.assign(X=x_data).dropna(subset=['X'])
        model = smf.logit(formula, data=tdf).fit(disp=0)
        ci = np.exp(model.conf_int().loc['X'])
        results.append({
            'Model': name,
            'AIC': model.aic,
            'P': model.pvalues['X'],
            'OR': np.exp(model.params['X']),
            'CI_LO': float(ci.iloc[0]),
            'CI_HI': float(ci.iloc[1]),
            'CI': f"{ci.iloc[0]:.2f}-{ci.iloc[1]:.2f}",
        })

    res_df = pd.DataFrame(results).sort_values('AIC')

    # --- 6. 结果可视化模块 ---
    print_status("生成可视化图表...")
    fig_geno = plot_genotype_distribution(df_final)
    fig_age  = plot_age_boxplot(df_final, col_map['AGE'], p_age)
    fig_hwe  = plot_hwe(obs, exp, hwe_p)
    fig_forest = plot_forest(res_df)
    print_status(f"图表已保存至 {CONFIG['FIGURE_DIR']}/ 目录")

    # --- 7. 生成报告 ---
    with open(CONFIG["OUTPUT_MD"], "w", encoding="utf-8") as f:
        f.write(f"# 精神分裂症 RS2071313 统计报告\n\n")
        f.write(f"## 1. 基线资料\n")
        f.write(f"- 性别 P={p_sex:.4f} | 年龄 P={p_age:.4f} "
                f"(M(P25,P75): {age_ctrl.median():.1f} vs {age_case.median():.1f})\n")
        f.write(f"- 年龄 IQR 法异常值: {n_age_outliers} 例（已标注，未剔除）\n\n")
        f.write(f"![年龄分布]({fig_age})\n\n")
        f.write(f"## 2. 基因型分布\n\n")
        f.write(f"![基因型分布]({fig_geno})\n\n")
        f.write(f"## 3. HWE 检验\n- P = {hwe_p:.4f} "
                f"({'平衡' if hwe_p > 0.05 else '不平衡'})\n\n")
        f.write(f"![HWE 检验]({fig_hwe})\n\n")
        f.write(f"## 4. Logistic 回归 (最优模型: {res_df.iloc[0]['Model']})\n\n")
        f.write(res_df[['Model', 'AIC', 'P', 'OR', 'CI']].to_markdown(index=False))
        f.write(f"\n\n![Forest Plot]({fig_forest})\n")

    print_status(f"全部完成！报告见 {CONFIG['OUTPUT_MD']}")

except Exception as e:
    print_status(f"流程中断：{e}")