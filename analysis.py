import pandas as pd
import numpy as np
import pyreadstat
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import glob
import re
import os

CONFIG = {
    "REF_GENO": "GG",                      
    "OUTPUT_MD": "Schizo_Association_Final_Report.md"
}

def print_status(msg):
    print(f">>> [执行中] {msg}", flush=True)

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
        results.append({'Model': name, 'AIC': model.aic, 'P': model.pvalues['X'], 'OR': np.exp(model.params['X']), 'CI': f"{ci[0]:.2f}-{ci[1]:.2f}"})

    res_df = pd.DataFrame(results).sort_values('AIC')
    
    # --- 6. 生成报告 ---
    with open(CONFIG["OUTPUT_MD"], "w", encoding="utf-8") as f:
        f.write(f"# 精神分裂症 RS2071313 统计报告\n\n")
        f.write(f"## 1. 基线资料\n")
        f.write(f"- 性别 P={p_sex:.4f} | 年龄 P={p_age:.4f} (M(P25,P75): {age_ctrl.median():.1f} vs {age_case.median():.1f})\n\n")
        f.write(f"## 2. HWE 检验\n- P = {hwe_p:.4f} ({'平衡' if hwe_p>0.05 else '不平衡'})\n\n")
        f.write(f"## 3. Logistic 回归 (最优模型: {res_df.iloc[0]['Model']})\n\n")
        f.write(res_df.to_markdown(index=False))

    print_status(f"全部完成！报告见 {CONFIG['OUTPUT_MD']}")

except Exception as e:
    print_status(f"流程中断：{e}")