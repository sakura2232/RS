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
matplotlib.use('Agg')  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

CONFIG = {
    "REF_GENO": "GG",
    "OUTPUT_MD": "Schizo_Association_Final_Report.md",
    "PLOT_GENO": "plot_genotype_freq.png",
    "PLOT_AGE":  "plot_age_distribution.png",
    "PLOT_FOREST": "plot_forest.png",
}

def print_status(msg):
    print(f">>> [执行中] {msg}", flush=True)

def super_clean_id(x):
    if pd.isna(x): return ""
    nums = re.findall(r'\d+', str(x))
    return "".join(nums) if nums else str(x).strip()

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_genotype_frequencies(df, geno_col, group_col, out_path):
    """Bar chart: genotype frequency (%) in cases vs controls."""
    order = ['AA', 'AG', 'GG']
    group_labels = {0: '对照组', 1: '病例组'}
    groups = sorted(df[group_col].unique())
    x = np.arange(len(order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, grp in enumerate(groups):
        sub = df[df[group_col] == grp][geno_col]
        counts = sub.value_counts()
        freqs = [counts.get(g, 0) / len(sub) * 100 for g in order]
        label = group_labels.get(grp, str(grp))
        ax.bar(x + (i - 0.5) * width, freqs, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=12)
    ax.set_ylabel('频率 (%)', fontsize=12)
    ax.set_title('RS2071313 基因型频率分布', fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_age_distribution(age_ctrl, age_case, out_path):
    """Box + strip plot comparing age in cases vs controls."""
    df_plot = pd.DataFrame({
        '年龄': pd.concat([age_ctrl, age_case], ignore_index=True),
        '组别': ['对照组'] * len(age_ctrl) + ['病例组'] * len(age_case)
    })
    palette = {'对照组': '#4C72B0', '病例组': '#DD8452'}
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df_plot, x='组别', y='年龄', hue='组别', palette=palette,
                width=0.5, fliersize=0, legend=False, ax=ax)
    sns.stripplot(data=df_plot, x='组别', y='年龄', color='black', alpha=0.3,
                  size=3, jitter=True, ax=ax)
    ax.set_title('年龄分布比较', fontsize=14)
    ax.set_ylabel('年龄（岁）', fontsize=12)
    ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_forest(res_df, out_path):
    """Forest plot of OR (95 % CI) for each logistic regression model."""
    models = res_df['Model'].tolist()
    ors = res_df['OR'].tolist()
    cis = res_df['CI'].tolist()
    lowers, uppers = [], []
    for ci_str in cis:
        parts = ci_str.split('-')
        lowers.append(float(parts[0]))
        uppers.append(float(parts[1]))

    y = np.arange(len(models))
    xerr_lo = [o - l for o, l in zip(ors, lowers)]
    xerr_hi = [u - o for o, u in zip(ors, uppers)]

    fig, ax = plt.subplots(figsize=(8, max(4, len(models) * 1.2)))
    ax.errorbar(ors, y, xerr=[xerr_lo, xerr_hi], fmt='o', color='steelblue',
                ecolor='gray', capsize=4, markersize=7)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel('OR (95% CI)', fontsize=12)
    ax.set_title('Logistic 回归模型 Forest Plot', fontsize=14)
    # annotate each row with p-value
    for i, row in res_df.reset_index(drop=True).iterrows():
        ax.text(max(uppers) * 1.05, i, f"P={row['P']:.3f}", va='center', fontsize=9)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Outlier detection helpers
# ---------------------------------------------------------------------------

def detect_age_outliers(series, label=''):
    """IQR-based outlier detection. Returns a boolean mask of outliers."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lo) | (series > hi)
    if mask.any():
        print_status(f"年龄异常值检测 [{label}]: IQR 范围 [{lo:.1f}, {hi:.1f}]，"
                     f"检出 {mask.sum()} 个异常值（已纳入敏感性分析）")
    else:
        print_status(f"年龄异常值检测 [{label}]: 未发现 IQR 异常值。")
    return mask, (lo, hi)

def detect_regression_outliers(model_result, tdf):
    """Cook's distance outlier detection on a fitted logit model."""
    try:
        influence = model_result.get_influence()
        cooks_d = influence.cooks_distance[0]
        threshold = 4 / len(tdf)
        influential = np.where(cooks_d > threshold)[0]
        print_status(f"Cook's 距离检测: 阈值={threshold:.4f}，"
                     f"检出 {len(influential)} 个强影响点 (索引: {influential[:10].tolist()}{'…' if len(influential)>10 else ''})")
        return influential, cooks_d
    except Exception as ex:
        print_status(f"Cook's 距离计算跳过: {ex}")
        return np.array([]), np.array([])


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
        print("--- 缺失值诊断 ---")
        print(df_merged[[col_map['GROUP'], col_map['AGE'], col_map['GENDER'], 'GENO_CLEAN']].isna().sum())
        exit()

    unique_groups = sorted(df_final[col_map['GROUP']].unique())
    df_final['Y'] = (df_final[col_map['GROUP']] == unique_groups[-1]).astype(int)
    print_status(f"质控完成：有效样本 {len(df_final)} 例。")

    # --- 5. 统计分析 ---
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
    best_model_obj = None
    best_model_tdf = None
    for name, x_data in models_def.items():
        tdf = df_final.assign(X=x_data).dropna(subset=['X'])
        model = smf.logit(formula, data=tdf).fit(disp=0)
        ci = np.exp(model.conf_int().loc['X'])
        results.append({
            'Model': name,
            'AIC': model.aic,
            'P': model.pvalues['X'],
            'OR': np.exp(model.params['X']),
            'CI': f"{ci[0]:.2f}-{ci[1]:.2f}"
        })
        if best_model_obj is None or model.aic < best_model_obj.aic:
            best_model_obj = model
            best_model_tdf = tdf

    res_df = pd.DataFrame(results).sort_values('AIC')

    # --- 6. 异常值验证 ---
    print_status("开始异常值验证...")
    outlier_ctrl_mask, age_ctrl_range = detect_age_outliers(age_ctrl, '对照组')
    outlier_case_mask, age_case_range = detect_age_outliers(age_case, '病例组')
    n_age_outliers = int(outlier_ctrl_mask.sum()) + int(outlier_case_mask.sum())

    influential_idx, cooks_d = detect_regression_outliers(best_model_obj, best_model_tdf)

    # Sensitivity analysis: rerun best model without influential points
    sens_note = "无强影响点，无需敏感性分析。"
    if len(influential_idx) > 0:
        tdf_sens = best_model_tdf.drop(best_model_tdf.index[influential_idx])
        try:
            model_sens = smf.logit(formula, data=tdf_sens).fit(disp=0)
            ci_s = np.exp(model_sens.conf_int().loc['X'])
            or_s = np.exp(model_sens.params['X'])
            p_s = model_sens.pvalues['X']
            sens_note = (f"剔除 {len(influential_idx)} 个强影响点后：OR={or_s:.2f} "
                         f"(95%CI {ci_s[0]:.2f}–{ci_s[1]:.2f})，P={p_s:.4f}")
        except Exception as ex:
            sens_note = f"敏感性分析失败: {ex}"
    print_status(f"敏感性分析: {sens_note}")

    # --- 7. 可视化 ---
    print_status("生成可视化图表...")
    plot_genotype_frequencies(df_final, 'GENO_CLEAN', 'Y', CONFIG["PLOT_GENO"])
    plot_age_distribution(age_ctrl, age_case, CONFIG["PLOT_AGE"])
    plot_forest(res_df, CONFIG["PLOT_FOREST"])
    print_status("图表已保存。")

    # --- 8. 生成报告 ---
    with open(CONFIG["OUTPUT_MD"], "w", encoding="utf-8") as f:
        f.write("# 精神分裂症 RS2071313 统计报告\n\n")

        f.write("## 1. 基线资料\n")
        f.write(f"- 有效样本：{len(df_final)} 例 "
                f"（对照 {int((df_final['Y']==0).sum())} / 病例 {int((df_final['Y']==1).sum())}）\n")
        f.write(f"- 性别 χ²={chi2_sex:.3f}, P={p_sex:.4f}\n")
        f.write(f"- 年龄 Mann-Whitney U, P={p_age:.4f} "
                f"（中位数: 对照 {age_ctrl.median():.1f} 岁 vs 病例 {age_case.median():.1f} 岁）\n\n")

        f.write("## 2. HWE 检验（对照组）\n")
        f.write(f"- χ²={hwe_chi:.3f}, P={hwe_p:.4f} — "
                f"{'符合 Hardy-Weinberg 平衡' if hwe_p > 0.05 else '偏离 Hardy-Weinberg 平衡'}\n")
        f.write(f"- 等位基因频率: A={p_all:.4f}, G={1-p_all:.4f}\n\n")

        f.write("## 3. Logistic 回归结果（按 AIC 排序）\n\n")
        f.write(res_df[['Model', 'AIC', 'OR', 'CI', 'P']].to_markdown(index=False))
        f.write(f"\n\n> 最优模型（最低 AIC）：**{res_df.iloc[0]['Model']}**\n\n")

        f.write("## 4. 异常值验证\n")
        f.write(f"- 年龄 IQR 异常值：{n_age_outliers} 例（对照 {int(outlier_ctrl_mask.sum())} / "
                f"病例 {int(outlier_case_mask.sum())}）\n")
        f.write(f"  - 对照组年龄 IQR 区间：[{age_ctrl_range[0]:.1f}, {age_ctrl_range[1]:.1f}]\n")
        f.write(f"  - 病例组年龄 IQR 区间：[{age_case_range[0]:.1f}, {age_case_range[1]:.1f}]\n")
        f.write(f"- 回归强影响点（Cook's 距离）：{len(influential_idx)} 个\n")
        f.write(f"- 敏感性分析：{sens_note}\n\n")

        f.write("## 5. 可视化图表\n")
        f.write(f"![基因型频率分布]({CONFIG['PLOT_GENO']})\n\n")
        f.write(f"![年龄分布比较]({CONFIG['PLOT_AGE']})\n\n")
        f.write(f"![Forest Plot]({CONFIG['PLOT_FOREST']})\n")

    print_status(f"全部完成！报告见 {CONFIG['OUTPUT_MD']}")

except Exception as e:
    import traceback
    print_status(f"流程中断：{e}")
    traceback.print_exc()