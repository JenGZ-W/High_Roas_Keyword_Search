import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
from datetime import timedelta
import io

# ==========================================
# 0. 页面全局配置 - 专业决策命名
# ==========================================
st.set_page_config(page_title="Walmart广告深度诊断决策系统", layout="wide")
st.title("🛡️ Walmart 广告深度诊断决策系统")
st.markdown("---")

# ==========================================
# 1. 核心算法逻辑 (非必要不删改)
# ==========================================
st.sidebar.header("🎯 决策参数配置")
ROAS_HIGH = st.sidebar.number_input("高转化 ROAS 阈值", value=3.5, step=0.1)
ROAS_LOW = st.sidebar.number_input("烧钱词 ROAS 预警线", value=2.0, step=0.1)
SPEND_MONEY_PIT = st.sidebar.number_input("烧钱词判定消耗 ($)", value=30.0, step=5.0)
MIN_CLICKS_FOR_VALID = st.sidebar.number_input("最小点击量门槛", value=5, step=1)

COL_DATE, COL_ITEM_ID, COL_SPEND = 'Date', 'Item Id', 'Ad Spend'
COL_SALES, COL_CLICKS, COL_ATC = 'Total Attributed Sales', 'Clicks', 'Total Add to Cart'

def check_atc_high(product_type, atc_rate):
    if pd.isna(product_type): return atc_rate >= 0.07
    pt = str(product_type).strip().lower()
    if pt in ['dresses', 'outfit sets']: return atc_rate >= 0.03
    elif pt in ['jumpsuits', 'skirts']: return atc_rate >= 0.05
    else: return atc_rate >= 0.07

def evaluate_metrics(sales, spend, clicks, atc, product_type):
    roas = sales / spend if spend > 0 else 0
    atc_rate = atc / clicks if clicks > 0 else 0
    is_hc = (roas >= ROAS_HIGH) and (clicks >= MIN_CLICKS_FOR_VALID)
    is_ha = check_atc_high(product_type, atc_rate) and (clicks >= MIN_CLICKS_FOR_VALID)
    is_mp = (roas < ROAS_LOW) and (spend >= SPEND_MONEY_PIT)
    return roas, is_hc, is_ha, is_mp

def get_action_suggestion(match_name, data, product_type, mp_text):
    sales, spend, clicks, atc = data
    if spend == 0 and clicks == 0: return None
    roas, hc, ha, mp = evaluate_metrics(sales, spend, clicks, atc, product_type)
    if hc: return f"开启 {match_name}(高转化)"
    elif ha:
        if roas < ROAS_LOW: return f"需调整 {match_name}(高加购低转化)"
        else: return f"开启 {match_name}(高加购)"
    elif mp: return mp_text
    return None

@st.cache_data
def build_category_gold_pool(df):
    pool = {} 
    category_broad_fingerprints = {} 
    cat_group = df.groupby(['Product Type', 'Standard_Keyword']).agg({
        COL_SALES: 'sum', COL_SPEND: 'sum', COL_CLICKS: 'sum', COL_ATC: 'sum'
    }).reset_index()
    
    for _, row in cat_group.iterrows():
        if row[COL_SPEND] <= 0: continue
        fingerprint = frozenset(str(row['Standard_Keyword']).split())
        cat = row['Product Type']
        if cat not in category_broad_fingerprints: category_broad_fingerprints[cat] = set()
        if fingerprint in category_broad_fingerprints[cat]: continue
            
        roas, is_hc, is_ha, _ = evaluate_metrics(row[COL_SALES], row[COL_SPEND], row[COL_CLICKS], row[COL_ATC], row['Product Type'])
        reason = ""
        if is_hc and is_ha: reason = "高转化且高加购"
        elif is_hc: reason = "高转化"
        elif is_ha: reason = "高加购"
        
        if reason:
            if cat not in pool: pool[cat] = {}
            pool[cat][row['Standard_Keyword']] = (reason, row[COL_SPEND])
            category_broad_fingerprints[cat].add(fingerprint)
    return pool

@st.cache_data
def run_dual_track_analysis(df_period, gold_pool, period_label):
    results = []
    listings = df_period['Listing'].dropna().unique()
    for listing in listings:
        df_list = df_period[df_period['Listing'] == listing]
        product_type = df_list['Product Type'].iloc[0]
        existing_keywords = df_list['Standard_Keyword'].dropna().unique()
        cat_gold_dict = gold_pool.get(product_type, {})
        all_keywords_to_check = set(existing_keywords) | set(cat_gold_dict.keys())
        evaluated_broad_sets = set()
        
        for target_x in sorted(list(all_keywords_to_check), key=lambda x: len(str(x))):
            if str(target_x).lower() in ['invalid', 'nan'] or len(str(target_x)) < 3: continue
            words_in_x = str(target_x).split()
            broad_fingerprint = frozenset(words_in_x)
            
            exact_mask = (df_list['Standard_Keyword'] == target_x)
            phrase_pattern = r'\b' + re.escape(str(target_x)) + r'\b'
            phrase_mask = df_list['Standard_Keyword'].str.contains(phrase_pattern, regex=True, na=False)
            
            p_ex, p_ph = df_list[exact_mask], df_list[phrase_mask & ~exact_mask]
            ex_data = [p_ex[COL_SALES].sum(), p_ex[COL_SPEND].sum(), p_ex[COL_CLICKS].sum(), p_ex[COL_ATC].sum()]
            ph_data = [p_ph[COL_SALES].sum(), p_ph[COL_SPEND].sum(), p_ph[COL_CLICKS].sum(), p_ph[COL_ATC].sum()]
            
            is_duplicate_broad = False
            if broad_fingerprint in evaluated_broad_sets:
                br_data, is_duplicate_broad = [0.0, 0.0, 0.0, 0.0], True
            else:
                broad_mask = df_list['Standard_Keyword'].apply(lambda x: all(w in str(x).split() for w in words_in_x))
                p_br = df_list[broad_mask & ~phrase_mask]
                br_data = [p_br[COL_SALES].sum(), p_br[COL_SPEND].sum(), p_br[COL_CLICKS].sum(), p_br[COL_ATC].sum()]
                evaluated_broad_sets.add(broad_fingerprint)
            
            total_spend = sum([ex_data[1], ph_data[1], br_data[1]])
            if total_spend == 0 and is_duplicate_broad: continue
                
            recs, remark = [], ""
            if total_spend > 0:
                ae = get_action_suggestion("精确", ex_data, product_type, "关停精确(烧钱)")
                if ae: recs.append(ae)
                ap = get_action_suggestion("词组", ph_data, product_type, "否定词组延伸(烧钱)")
                if ap: recs.append(ap)
                ab = get_action_suggestion("广泛", br_data, product_type, "否定广泛延伸(烧钱)")
                if ab: recs.append(ab)
                if target_x in cat_gold_dict:
                    reason, cat_spend = cat_gold_dict[target_x]
                    remark = f"✔️ 属品类【{reason}】共性词 (大盘耗: ${cat_spend:.2f})"
            elif target_x in cat_gold_dict:
                reason, cat_spend = cat_gold_dict[target_x]
                recs.append(f"建议测试 词组/广泛 (依据:大盘{reason})")
                remark = f"🌟 品类潜力词推荐 (大盘已验证耗: ${cat_spend:.2f})"
            
            if recs:
                # 【优化重点】：补充加购数量和加购率展示
                results.append({
                    'Time_Window': period_label, 'Listing': listing, 'Product_Type': product_type, 'Keyword_X': target_x,
                    'Action_Suggestion': " | ".join(recs), 'Special_Remark': remark,
                    'Total_Spend': total_spend, 
                    'Overall_ROAS': (ex_data[0]+ph_data[0]+br_data[0])/total_spend if total_spend > 0 else 0,
                    'Overall_ATC%': (ex_data[3]+ph_data[3]+br_data[3])/(ex_data[2]+ph_data[2]+br_data[2]) if (ex_data[2]+ph_data[2]+br_data[2]) > 0 else 0,
                    'Ex_Spend': ex_data[1], 'Ex_ROAS': ex_data[0]/ex_data[1] if ex_data[1] > 0 else 0, 'Ex_ATC%': ex_data[3]/ex_data[2] if ex_data[2]>0 else 0,
                    'Ph_Spend': ph_data[1], 'Ph_ROAS': ph_data[0]/ph_data[1] if ph_data[1] > 0 else 0, 'Ph_ATC%': ph_data[3]/ph_data[2] if ph_data[2]>0 else 0,
                    'Br_Spend': br_data[1], 'Br_ROAS': br_data[0]/br_data[1] if br_data[1] > 0 else 0, 'Br_ATC%': br_data[3]/br_data[2] if br_data[2]>0 else 0,
                })
    return pd.DataFrame(results)

# ==========================================
# 2. 全局联动筛选 (Sidebar)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("📁 数据源加载")
f1 = st.sidebar.file_uploader("1. 原始报表", type=['csv', 'xlsx'])
f2 = st.sidebar.file_uploader("2. 标准词典", type=['csv', 'xlsx'])
f3 = st.sidebar.file_uploader("3. 产品基础信息", type=['csv', 'xlsx'])

if f1 and f2 and f3:
    d_main = pd.read_csv(f1) if f1.name.endswith('csv') else pd.read_excel(f1)
    d_ai = pd.read_csv(f2) if f2.name.endswith('csv') else pd.read_excel(f2)
    d_base = pd.read_csv(f3) if f3.name.endswith('csv') else pd.read_excel(f3)

    d_main[COL_ITEM_ID], d_base[COL_ITEM_ID] = d_main[COL_ITEM_ID].astype(str), d_base[COL_ITEM_ID].astype(str)
    d_main = pd.merge(d_main, d_ai[['Cleaned_Keyword', 'Standard_Keyword']], on='Cleaned_Keyword', how='left')
    d_main = pd.merge(d_main, d_base[['Item Id', 'Listing', 'Product Type']], on=COL_ITEM_ID, how='left')
    d_main['Standard_Keyword'] = d_main['Standard_Keyword'].fillna(d_main['Cleaned_Keyword']).astype(str).str.lower()
    d_main[COL_DATE] = pd.to_datetime(d_main[COL_DATE])
    mx_dt = d_main[COL_DATE].max()

    g_pool = build_category_gold_pool(d_main)
    
    # 异步生成全量建议
    r14 = run_dual_track_analysis(d_main[d_main[COL_DATE] >= (mx_dt - timedelta(days=14))], g_pool, '14日执行建议')
    r30 = run_dual_track_analysis(d_main[d_main[COL_DATE] >= (mx_dt - timedelta(days=30))], g_pool, '30日策略分析')
    r60 = run_dual_track_analysis(d_main[d_main[COL_DATE] >= (mx_dt - timedelta(days=60))], g_pool, '60日趋势透视')
    full_df = pd.concat([r14, r30, r60], ignore_index=True)

    # 全局过滤器
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 全域动态筛选")
    full_df['Listing'] = full_df['Listing'].astype(str)
    
    sel_types = st.sidebar.multiselect("品类筛选", sorted(full_df['Product_Type'].dropna().unique()), default=full_df['Product_Type'].unique())
    sel_listings = st.sidebar.multiselect("Listing筛选", sorted(full_df['Listing'].dropna().unique()), default=full_df['Listing'].unique())
    sel_actions = st.sidebar.multiselect("动作动机", ["进攻 (开启/测试)", "调控 (需调整)", "防守 (关停/否定)"], default=["进攻 (开启/测试)", "调控 (需调整)", "防守 (关停/否定)"])
    search_kw = st.sidebar.text_input("搜标准词", "").strip().lower()

    def apply_filter(df):
        df['Listing'] = df['Listing'].astype(str)
        temp = df[df['Product_Type'].isin(sel_types) & df['Listing'].isin(sel_listings)]
        if search_kw: temp = temp[temp['Keyword_X'].str.contains(search_kw)]
        mask = pd.Series([False] * len(temp), index=temp.index)
        if "进攻 (开启/测试)" in sel_actions: mask |= temp['Action_Suggestion'].str.contains("开启|建议测试")
        if "调控 (需调整)" in sel_actions: mask |= temp['Action_Suggestion'].str.contains("需调整")
        if "防守 (关停/否定)" in sel_actions: mask |= temp['Action_Suggestion'].str.contains("关停|否定")
        return temp[mask]

    # ==========================================
    # 3. 前端可视化渲染 (优化可视性)
    # ==========================================
    f_res_14, f_res_30, f_res_60 = apply_filter(r14), apply_filter(r30), apply_filter(r60)
    f_full = pd.concat([f_res_14, f_res_30, f_res_60])

    st.success(f"📊 诊断报告生成成功 | 全局匹配策略 {len(f_full)} 条")
    
    tabs = st.tabs(["🔭 趋势透视", "⚡ 14D 紧急策略", "📈 30D 周期诊断", "🗺️ 60D 全局概览", "💎 潜力词挖矿"])
    
    with tabs[0]:
        bubble_data = f_full[f_full['Total_Spend'] > 0]
        if not bubble_data.empty:
            fig = px.scatter(bubble_data, x="Total_Spend", y="Overall_ROAS", color="Product_Type",
                             size="Overall_ATC%", hover_name="Keyword_X",
                             labels={"Total_Spend": "消耗", "Overall_ROAS": "ROAS", "Overall_ATC%": "加购率"},
                             title="消耗/转化关系图 (气泡大小代表加购率强度)")
            fig.add_hline(y=ROAS_LOW, line_dash="dot", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

    # 【修复】：增加 df.reset_index(drop=True) 解决非唯一索引报错
    def highlight_dataframe(df):
        if df.empty: return df
        df = df.reset_index(drop=True)  # <--- 增加这一行：重置行号，确保唯一性
        
        # 仅保留操作指令的红绿底色，去除数值列的背景渐变 (适配你的无 matplotlib 环境)
        return df.style.map(lambda val: 'background-color: #e6ffed; color: #155724; font-weight: bold;' if any(x in str(val) for x in ['开启', '建议测试'])
                                 else ('background-color: #fff1f0; color: #cf1322; font-weight: bold;' if any(x in str(val) for x in ['否定', '关停']) 
                                       else ('background-color: #fffbe6; color: #856404;' if '需调整' in str(val) else '')), 
                            subset=['Action_Suggestion'])\
                       .format(subset=['Overall_ATC%', 'Ex_ATC%', 'Ph_ATC%', 'Br_ATC%'], formatter='{:.2%}')\
                       .format(subset=['Total_Spend', 'Overall_ROAS', 'Ex_Spend', 'Ex_ROAS', 'Ph_Spend', 'Ph_ROAS', 'Br_Spend', 'Br_ROAS'], formatter='{:.2f}')

    # 在渲染时使用更加清晰的配置
    col_cfg = {
        "Overall_ROAS": st.column_config.NumberColumn("总体 ROAS", format="%.2f"),
        "Overall_ATC%": st.column_config.ProgressColumn("总体加购率", format="%.2f%%", min_value=0, max_value=0.2),
        "Action_Suggestion": "操作建议"
    }

    with tabs[1]: st.dataframe(highlight_dataframe(f_res_14), use_container_width=True, column_config=col_cfg)
    with tabs[2]: st.dataframe(highlight_dataframe(f_res_30), use_container_width=True, column_config=col_cfg)
    with tabs[3]: st.dataframe(highlight_dataframe(f_res_60), use_container_width=True, column_config=col_cfg)
    with tabs[4]:
        pot = f_full[f_full['Special_Remark'].str.contains('潜力词推荐', na=False)].drop_duplicates(subset=['Listing', 'Keyword_X'])
        st.dataframe(highlight_dataframe(pot), use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as wr:
        f_full.to_excel(wr, sheet_name='决策报告', index=False)
    st.download_button("💾 导出带加购数据的深度诊断报告", data=buf.getvalue(), file_name="Walmart_Deep_Strategy_Report.xlsx")
else:
    st.info("💡 请在侧边栏上传 Phase1_Cleaned, Standard_Keywords, Base_Info 三个核心文件。")