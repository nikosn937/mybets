import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import math
import re
import time
from google import genai
from google.genai import types

st.set_page_config(page_title="Pro Football Predictor + Combo Market Generator", layout="wide")

st.title("⚽ Pro Football Predictor (Dixon-Coles + Combo Market Generator)")
st.subheader("Σεζόν 2026/2027 | Cross-League Multi-Market Scanner & Combo Betting")

SEASON_CODE = "2627"

LEAGUES = {
    "Premier League (Αγγλία)": {
        "history": f"https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/E0.csv",
        "code": "E0"
    },
    "La Liga (Ισπανία)": {
        "history": f"https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/SP1.csv",
        "code": "SP1"
    },
    "Bundesliga (Γερμανία)": {
        "history": f"https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/D1.csv",
        "code": "D1"
    },
    "Serie A (Ιταλία)": {
        "history": f"https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/I1.csv",
        "code": "I1"
    },
    "Super League (Ελλάδα)": {
        "history": f"https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/G1.csv",
        "code": "G1"
    }
}

# Sidebar Παράμετροι
st.sidebar.header("⚙️ Παράμετροι Μοντέλου")
selected_league_name = st.sidebar.selectbox("Επιλέξτε Πρωτάθλημα για Αναλυτική Προβολή", list(LEAGUES.keys()))

CONFIDENCE_THRESHOLD = st.sidebar.slider("Ελάχιστο Ποσοστό Σιγουριάς (%)", min_value=50, max_value=90, value=60, step=5)
USE_WEIGHTS = st.sidebar.checkbox("Στάθμιση Φόρμας (Exponential Time-Decay)", value=True)
DECAY_XI = st.sidebar.slider("Ρυθμός Απόσβεσης Χρόνου (Xi)", min_value=0.001, max_value=0.015, value=0.005, step=0.001, help="Υψηλότερη τιμή δίνει μεγαλύτερη έμφαση στα πολύ πρόσφατα παιχνίδια.")
USE_H2H = st.sidebar.checkbox("Ενεργοποίηση Προσαρμογής H2H (10ετής Προϊστορία)", value=True)
RHO_DIXON = st.sidebar.slider("Συντελεστής Dixon-Coles (Rho)", min_value=-0.25, max_value=0.0, value=-0.13, step=0.01)

def clean_team_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9]', '', name)
    replacements = {
        "manchesterunited": "manunited",
        "manchester city": "mancity",
        "nottinghamforest": "nottmforest"
    }
    return replacements.get(name, name)

@st.cache_data(ttl=1800)
def load_history_data(url):
    try:
        df = pd.read_csv(url)
        cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        df = df[cols].dropna()
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        return df.dropna(subset=['Date']).sort_values('Date')
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_multi_season_h2h(league_code):
    seasons = [
        "2627", "2526", "2425", "2324", "2223", 
        "2122", "2021", "1920", "1819", "1718"
    ]
    dfs = []
    for s in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{s}/{league_code}.csv"
        try:
            df = pd.read_csv(url)
            cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            df = df[cols].dropna()
            dfs.append(df)
        except Exception:
            continue
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

@st.cache_data(ttl=1800)
def load_fixtures_data(code):
    try:
        url = "https://www.football-data.co.uk/fixtures.csv"
        df = pd.read_csv(url)
        df = df[df['Div'] == code]
        cols = ['Date', 'Time', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols].dropna(subset=['HomeTeam', 'AwayTeam'])
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        return df.dropna(subset=['Date'])
    except Exception:
        return None

def calculate_h2h_adjustment(df_h2h_all, home_team, away_team, max_matches=20):
    if df_h2h_all is None or df_h2h_all.empty:
        return 1.0, 1.0, "—"
    clean_home = clean_team_name(home_team)
    clean_away = clean_team_name(away_team)
    df = df_h2h_all.copy()
    df['c_home'] = df['HomeTeam'].apply(clean_team_name)
    df['c_away'] = df['AwayTeam'].apply(clean_team_name)
    h2h_matches = df[
        ((df['c_home'] == clean_home) & (df['c_away'] == clean_away)) |
        ((df['c_home'] == clean_away) & (df['c_away'] == clean_home))
    ].tail(max_matches)
    total_games = len(h2h_matches)
    if total_games == 0:
        return 1.0, 1.0, "Χωρίς H2H"
    home_wins, away_wins, draws = 0, 0, 0
    for _, row in h2h_matches.iterrows():
        if row['FTHG'] > row['FTAG']:
            if row['c_home'] == clean_home: home_wins += 1
            else: away_wins += 1
        elif row['FTAG'] > row['FTHG']:
            if row['c_away'] == clean_away: away_wins += 1
            else: home_wins += 1
        else:
            draws += 1
    home_ratio = home_wins / total_games
    away_ratio = away_wins / total_games
    h2h_home_mult = float(np.clip(1.0 + (home_ratio - 0.33) * 0.25, 0.90, 1.10))
    h2h_away_mult = float(np.clip(1.0 + (away_ratio - 0.33) * 0.25, 0.90, 1.10))
    summary_str = f"{home_wins}-{draws}-{away_wins} ({total_games} ματς)"
    return h2h_home_mult, h2h_away_mult, summary_str

def calculate_dixon_coles_stats(df, use_weights=True, xi=0.005):
    if use_weights and len(df) > 1:
        max_date = df['Date'].max()
        days_diff = (max_date - df['Date']).dt.days
        weights = np.exp(-xi * days_diff)
    else:
        weights = np.ones(len(df))
    avg_home_goals = np.average(df['FTHG'], weights=weights)
    avg_away_goals = np.average(df['FTAG'], weights=weights)
    teams = sorted(list(set(df['HomeTeam']).union(set(df['AwayTeam']))))
    stats = {}
    for team in teams:
        h_idx = df['HomeTeam'] == team
        a_idx = df['AwayTeam'] == team
        h_scored = np.average(df.loc[h_idx, 'FTHG'], weights=weights[h_idx]) if h_idx.sum() > 0 else avg_home_goals
        h_conceded = np.average(df.loc[h_idx, 'FTAG'], weights=weights[h_idx]) if h_idx.sum() > 0 else avg_away_goals
        a_scored = np.average(df.loc[a_idx, 'FTAG'], weights=weights[a_idx]) if a_idx.sum() > 0 else avg_away_goals
        a_conceded = np.average(df.loc[a_idx, 'FTHG'], weights=weights[a_idx]) if a_idx.sum() > 0 else avg_home_goals
        stats[team] = {
            'home_attack': h_scored / avg_home_goals if avg_home_goals > 0 else 1.0,
            'home_defense': h_conceded / avg_away_goals if avg_away_goals > 0 else 1.0,
            'away_attack': a_scored / avg_away_goals if avg_away_goals > 0 else 1.0,
            'away_defense': a_conceded / avg_home_goals if avg_home_goals > 0 else 1.0
        }
    return stats, avg_home_goals, avg_away_goals

def dixon_coles_adjustment(x, y, lambda_h, lambda_a, rho):
    if x == 0 and y == 0:
        return 1.0 - (lambda_h * lambda_a * rho)
    elif x == 0 and y == 1:
        return 1.0 + (lambda_h * rho)
    elif x == 1 and y == 0:
        return 1.0 + (lambda_a * rho)
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0

def predict_match_dc(home_team, away_team, stats, avg_h, avg_a, rho, h_adj=1.0, a_adj=1.0, h2h_h_mult=1.0, h2h_a_mult=1.0):
    default_stat = {'home_attack': 1, 'home_defense': 1, 'away_attack': 1, 'away_defense': 1}
    h_stat = stats.get(home_team, default_stat)
    a_stat = stats.get(away_team, default_stat)
    
    lambda_home = h_stat['home_attack'] * a_stat['away_defense'] * avg_h * h_adj * h2h_h_mult
    lambda_away = a_stat['away_attack'] * h_stat['home_defense'] * avg_a * a_adj * h2h_a_mult
    
    max_goals = 7
    score_matrix = np.zeros((max_goals, max_goals))
    for x in range(max_goals):
        for y in range(max_goals):
            p_x = poisson.pmf(x, lambda_home)
            p_y = poisson.pmf(y, lambda_away)
            adj = dixon_coles_adjustment(x, y, lambda_home, lambda_away, rho)
            score_matrix[x, y] = p_x * p_y * adj

    score_matrix = np.maximum(score_matrix, 0)
    score_matrix /= np.sum(score_matrix)
    
    prob_home = np.sum(np.tril(score_matrix, -1))
    prob_draw = np.sum(np.diag(score_matrix))
    prob_away = np.sum(np.triu(score_matrix, 1))
    
    goals_sum = np.add.outer(range(max_goals), range(max_goals))
    prob_over_1_5 = np.sum(score_matrix[goals_sum > 1.5])
    prob_under_1_5 = 1.0 - prob_over_1_5
    prob_over_2_5 = np.sum(score_matrix[goals_sum > 2.5])
    prob_under_2_5 = 1.0 - prob_over_2_5
    prob_over_3_5 = np.sum(score_matrix[goals_sum > 3.5])
    prob_under_3_5 = 1.0 - prob_over_3_5
    
    prob_btts_yes = np.sum(score_matrix[1:, 1:])
    prob_btts_no = 1.0 - prob_btts_yes
    
    tril_mask = np.tril(np.ones((max_goals, max_goals)), -1)
    triu_mask = np.triu(np.ones((max_goals, max_goals)), 1)
    
    prob_1_over_25 = np.sum(score_matrix[(goals_sum > 2.5) & (tril_mask == 1)])
    prob_2_over_25 = np.sum(score_matrix[(goals_sum > 2.5) & (triu_mask == 1)])
    prob_1_under_35 = np.sum(score_matrix[(goals_sum <= 3.5) & (tril_mask == 1)])
    prob_2_under_35 = np.sum(score_matrix[(goals_sum <= 3.5) & (triu_mask == 1)])
    
    not_triu_mask = 1.0 - np.triu(np.ones((max_goals, max_goals)), 1)
    not_tril_mask = 1.0 - np.tril(np.ones((max_goals, max_goals)), -1)
    prob_1x_over_15 = np.sum(score_matrix[(goals_sum > 1.5) & (not_triu_mask == 1)])
    prob_x2_over_15 = np.sum(score_matrix[(goals_sum > 1.5) & (not_tril_mask == 1)])

    score_tuples = []
    for x in range(max_goals):
        for y in range(max_goals):
            score_tuples.append((f"{x}-{y}", score_matrix[x, y]))
    score_tuples.sort(key=lambda item: item[1], reverse=True)
    
    top_score_name = score_tuples[0][0]
    top_score_prob = score_tuples[0][1]
    top_5_scores = score_tuples[:5]
    top_5_str_list = [f"{s[0]} ({round(s[1]*100, 1)}%)" for s in top_5_scores]
    top_5_cumulative = sum(s[1] for s in top_5_scores)
    conf_gap = top_5_scores[0][1] - top_5_scores[1][1]
    
    return {
        'xG_Home': round(lambda_home, 2),
        'xG_Away': round(lambda_away, 2),
        'Prob_1': prob_home,
        'Prob_X': prob_draw,
        'Prob_2': prob_away,
        'Prob_1X': prob_home + prob_draw,
        'Prob_X2': prob_away + prob_draw,
        'Prob_Over_1.5': prob_over_1_5,
        'Prob_Under_1.5': prob_under_1_5,
        'Prob_Over_2.5': prob_over_2_5,
        'Prob_Under_2.5': prob_under_2_5,
        'Prob_Over_3.5': prob_over_3_5,
        'Prob_Under_3.5': prob_under_3_5,
        'Prob_GG': prob_btts_yes,
        'Prob_NG': prob_btts_no,
        'Combo_1_Over_2.5': prob_1_over_25,
        'Combo_2_Over_2.5': prob_2_over_25,
        'Combo_1_Under_3.5': prob_1_under_35,
        'Combo_2_Under_3.5': prob_2_under_35,
        'Combo_1X_Over_1.5': prob_1x_over_15,
        'Combo_X2_Over_1.5': prob_x2_over_15,
        'Top_Score': top_score_name,
        'Top_Score_Prob': top_score_prob,
        'Top_5_Scores': ", ".join(top_5_str_list),
        'Top_5_Cumulative_%': round(top_5_cumulative * 100, 1),
        'Score_Confidence_Gap_%': round(conf_gap * 100, 1),
        'Score_Matrix': score_matrix
    }

@st.cache_data(ttl=3600, show_spinner=False)
def get_live_ai_analysis(home_team, away_team, date_str, stats_summary):
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return "⚠️ Δεν έχει οριστεί το GEMINI_API_KEY στα secrets του Streamlit."
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Είσαι ένας κορυφαίος αθλητικός αναλυτής και ποσοτικός αναλυτής στοιχημάτων (Quantitative Betting Analyst).
    Αντικείμενο: Αγώνας {home_team} vs {away_team} στις {date_str}.
    Δεδομένα Μοντέλου: {stats_summary}
    ΑΠΟΣΤΟΛΗ: Δώσε σύντομη αναφορά (3-4 bullet points) με τακτική αξιολόγηση xG, πιθανότερο σκορ και τελικό verdict. Γράψε στα Ελληνικά.
    """
    models_to_try = ['gemini-3.6-flash', 'gemini-1.5-flash']
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text
            except Exception as e:
                err_msg = str(e)
                if "503" in err_msg or "UNAVAILABLE" in err_msg:
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    return f"❌ Σφάλμα AI: {e}"
    return "⚠️ Οι διακομιστές της Google είναι προσωρινά μη διαθέσιμοι."

# --- ΦΙΛΤΡΟ ΗΜΕΡΟΜΗΝΙΩΝ ---
st.sidebar.divider()
st.sidebar.subheader("📅 Φίλτρο Ημερομηνιών")

init_fixtures = load_fixtures_data(LEAGUES[selected_league_name]["code"])
min_f_date = init_fixtures['Date'].min().date() if init_fixtures is not None and not init_fixtures.empty else pd.Timestamp.today().date()
max_f_date = init_fixtures['Date'].max().date() if init_fixtures is not None and not init_fixtures.empty else pd.Timestamp.today().date()

date_range = st.sidebar.date_input("Επιλέξτε εύρος ημερομηνιών", value=(min_f_date, max_f_date))
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
elif isinstance(date_range, tuple) and len(date_range) == 1:
    start_date, end_date = date_range[0], max_f_date
else:
    start_date, end_date = min_f_date, max_f_date

# --- ΣΑΡΩΣΗ ΟΛΩΝ ΤΩΝ ΠΡΩΤΑΘΛΗΜΑΤΩΝ ---
all_matches_list = []
league_predictions = {}
raw_predictions_dict = {}

for l_name, l_info in LEAGUES.items():
    l_code = l_info["code"]
    df_hist = load_history_data(l_info["history"])
    df_h2h = load_multi_season_h2h(l_code)
    df_fix = load_fixtures_data(l_code)
    
    if df_hist is not None and not df_hist.empty and df_fix is not None and not df_fix.empty:
        l_stats, l_avg_h, l_avg_a = calculate_dixon_coles_stats(df_hist, USE_WEIGHTS, DECAY_XI)
        filt_fix = df_fix[(df_fix['Date'].dt.date >= start_date) & (df_fix['Date'].dt.date <= end_date)]
        
        preds_list = []
        for idx, row in filt_fix.iterrows():
            h_team = row['HomeTeam']
            a_team = row['AwayTeam']
            m_date = row['Date'].strftime('%d/%m/%Y')
            m_time = row['Time'] if 'Time' in row and pd.notna(row['Time']) else ""
            match_name = f"{h_team} vs {a_team}"
            
            if USE_H2H:
                h2h_h_mult, h2h_a_mult, h2h_str = calculate_h2h_adjustment(df_h2h, h_team, a_team, max_matches=20)
            else:
                h2h_h_mult, h2h_a_mult, h2h_str = 1.0, 1.0, "Off"
                
            pred = predict_match_dc(h_team, a_team, l_stats, l_avg_h, l_avg_a, RHO_DIXON, 1.0, 1.0, h2h_h_mult, h2h_a_mult)
            raw_predictions_dict[match_name] = pred
            
            outcomes = [
                ("1", pred['Prob_1']), ("2", pred['Prob_2']), ("1X", pred['Prob_1X']), ("X2", pred['Prob_X2']),
                ("Over 1.5", pred['Prob_Over_1.5']), ("Under 1.5", pred['Prob_Under_1.5']),
                ("Over 2.5", pred['Prob_Over_2.5']), ("Under 2.5", pred['Prob_Under_2.5']),
                ("Over 3.5", pred['Prob_Over_3.5']), ("Under 3.5", pred['Prob_Under_3.5']),
                ("GG", pred['Prob_GG']), ("NG", pred['Prob_NG'])
            ]
            best_pick, best_prob = max(outcomes, key=lambda x: x[1])
            
            match_data = {
                "Πρωτάθλημα": l_name,
                "Ημερομηνία": m_date,
                "Ώρα": m_time,
                "Αγώνας": match_name,
                "Προτεινόμενο Σημείο": best_pick,
                "Πιθανότητα %": round(best_prob * 100, 1),
                "Over 1.5 %": round(pred['Prob_Over_1.5'] * 100, 1),
                "Under 1.5 %": round(pred['Prob_Under_1.5'] * 100, 1),
                "Over 2.5 %": round(pred['Prob_Over_2.5'] * 100, 1),
                "Under 2.5 %": round(pred['Prob_Under_2.5'] * 100, 1),
                "Over 3.5 %": round(pred['Prob_Over_3.5'] * 100, 1),
                "Under 3.5 %": round(pred['Prob_Under_3.5'] * 100, 1),
                "GG %": round(pred['Prob_GG'] * 100, 1),
                "1 & Over 2.5 %": round(pred['Combo_1_Over_2.5'] * 100, 1),
                "2 & Over 2.5 %": round(pred['Combo_2_Over_2.5'] * 100, 1),
                "1 & Under 3.5 %": round(pred['Combo_1_Under_3.5'] * 100, 1),
                "2 & Under 3.5 %": round(pred['Combo_2_Under_3.5'] * 100, 1),
                "1X & Over 1.5 %": round(pred['Combo_1X_Over_1.5'] * 100, 1),
                "X2 & Over 1.5 %": round(pred['Combo_X2_Over_1.5'] * 100, 1),
                "Πιθανότερο Σκορ": pred['Top_Score'],
                "Πιθανότητα Σκορ %": round(pred['Top_Score_Prob'] * 100, 1),
                "Top 5 Σκορ": pred['Top_5_Scores'],
                "Συγκέντρωση Top 5 %": pred['Top_5_Cumulative_%'],
                "Περιθώριο Σιγουριάς Σκορ %": pred['Score_Confidence_Gap_%'],
                "Προϊστορία (H2H)": h2h_str,
                "xG Γηπεδούχου": pred['xG_Home'],
                "xG Φιλοξενούμενου": pred['xG_Away']
            }
            preds_list.append(match_data)
            all_matches_list.append(match_data)
            
        league_predictions[l_name] = pd.DataFrame(preds_list)

st.info(f"📅 **Περίοδος Αγώνων: {start_date.strftime('%d/%m/%Y')} έως {end_date.strftime('%d/%m/%Y')}** ({len(all_matches_list)} συνολικοί αγώνες)")

# --- ΔΗΜΙΟΥΡΓΙΑ TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Top 20 Πιο Πιθανά Σκορ", 
    "🔀 Top 20 Combo Bets", 
    "⚽ Top 20 Over / Under", 
    f"📊 Αναλυτική Προβολή ({selected_league_name})"
])

# --- TAB 1: TOP 20 ΠΙΘΑΝΑ ΣΚΟΡ & DEEP MATRIX ANALYST ---
with tab1:
    st.subheader("🎯 Top 20 Πιο Πιθανά Ακριβή Σκορ (Όλα τα Πρωτάθληματα)")
    if all_matches_list:
        df_all = pd.DataFrame(all_matches_list)
        df_top_20_scores = df_all.sort_values(by="Πιθανότητα Σκορ %", ascending=False).head(20)
        display_cols_scores = ["Πρωτάθλημα", "Ημερομηνία", "Ώρα", "Αγώνας", "Πιθανότερο Σκορ", "Πιθανότητα Σκορ %", "Top 5 Σκορ", "Συγκέντρωση Top 5 %", "xG Γηπεδούχου", "xG Φιλοξενούμενου"]
        st.dataframe(df_top_20_scores[display_cols_scores], use_container_width=True)
        
        st.divider()
        st.subheader("🔎 Βαθιά Ανάλυση Matrix Σκορ Αγώνα (Heatmap 7x7)")
        selected_matrix_match = st.selectbox("Επιλέξτε αγώνα για ανάλυση Matrix Σκορ:", options=df_all['Αγώνας'].tolist())
        
        if selected_matrix_match in raw_predictions_dict:
            p_data = raw_predictions_dict[selected_matrix_match]
            matrix = p_data['Score_Matrix']
            col_m1, col_m2 = st.columns([3, 2])
            with col_m1:
                df_matrix = pd.DataFrame(np.round(matrix * 100, 2), index=[f"Home {i}" for i in range(7)], columns=[f"Away {j}" for j in range(7)])
                st.dataframe(df_matrix, use_container_width=True)
            with col_m2:
                st.info(f"""
                * **1ο Πιθανότερο Σκορ**: `{p_data['Top_Score']}` ({round(p_data['Top_Score_Prob']*100, 1)}%)
                * **Top 5 Σκορ**: {p_data['Top_5_Scores']}
                * **Συγκέντρωση Top 5**: `{p_data['Top_5_Cumulative_%']}%`
                * **Περιθώριο Σιγουριάς (Gap)**: `{p_data['Score_Confidence_Gap_%']}%`
                * **xG Αγώνα**: {p_data['xG_Home']} - {p_data['xG_Away']}
                """)
    else:
        st.warning("⚠️ Δεν βρέθηκαν επερχόμενοι αγώνες.")

# --- TAB 2: TOP 20 COMBO BETS ---
with tab2:
    st.subheader("🔀 Top 20 Συνδυαστικά Στοιχήματα (Combo Bets)")
    st.caption("Αυτόματος υπολογισμός πιθανοτήτων για συνδυασμούς σημείου και γκολ απευθείας από το Matrix.")
    
    if all_matches_list:
        df_all = pd.DataFrame(all_matches_list)
        
        sub_c1, sub_c2, sub_c3 = st.tabs([
            "🔥 Top 20 1/2 & Over 2.5", 
            "🛡️ Top 20 1/2 & Under 3.5", 
            "⚡ Top 20 1X/X2 & Over 1.5"
        ])
        
        c_cols = ["Πρωτάθλημα", "Ημερομηνία", "Ώρα", "Αγώνας", "xG Γηπεδούχου", "xG Φιλοξενούμενου"]
        
        with sub_c1:
            st.write("#### 🔥 Top 20 (1 ή 2) & Over 2.5")
            df_c1 = df_all.sort_values(by="1 & Over 2.5 %", ascending=False)
            df_c2 = df_all.sort_values(by="2 & Over 2.5 %", ascending=False)
            st.markdown("**Γηπεδούχος Νίκη & Over 2.5:**")
            st.dataframe(df_c1[c_cols + ["1 & Over 2.5 %"]].head(20), use_container_width=True)
            st.markdown("**Φιλοξενούμενη Νίκη & Over 2.5:**")
            st.dataframe(df_c2[c_cols + ["2 & Over 2.5 %"]].head(20), use_container_width=True)
            
        with sub_c2:
            st.write("#### 🛡️ Top 20 (1 ή 2) & Under 3.5 (Ασφαλή Combo)")
            df_u35_1 = df_all.sort_values(by="1 & Under 3.5 %", ascending=False)
            df_u35_2 = df_all.sort_values(by="2 & Under 3.5 %", ascending=False)
            st.markdown("**Γηπεδούχος Νίκη & Under 3.5:**")
            st.dataframe(df_u35_1[c_cols + ["1 & Under 3.5 %"]].head(20), use_container_width=True)
            st.markdown("**Φιλοξενούμενη Νίκη & Under 3.5:**")
            st.dataframe(df_u35_2[c_cols + ["2 & Under 3.5 %"]].head(20), use_container_width=True)
            
        with sub_c3:
            st.write("#### ⚡ Top 20 Διπλή Ευκαιρία & Over 1.5 (Η Απόλυτη Επιλογή Ασφαλείας)")
            df_1x = df_all.sort_values(by="1X & Over 1.5 %", ascending=False)
            df_x2 = df_all.sort_values(by="X2 & Over 1.5 %", ascending=False)
            st.markdown("**1Χ & Over 1.5:**")
            st.dataframe(df_1x[c_cols + ["1X & Over 1.5 %"]].head(20), use_container_width=True)
            st.markdown("**Χ2 & Over 1.5:**")
            st.dataframe(df_x2[c_cols + ["X2 & Over 1.5 %"]].head(20), use_container_width=True)
    else:
        st.warning("⚠️ Δεν βρέθηκαν επερχόμενοι αγώνες.")

# --- TAB 3: TOP 20 OVER / UNDER ---
with tab3:
    st.subheader("⚽ Top 20 Πιο Πιθανά Over / Under (Όλα τα Πρωτάθληματα)")
    if all_matches_list:
        df_all = pd.DataFrame(all_matches_list)
        subtab_o15, subtab_u15, subtab_o25, subtab_u25, subtab_o35, subtab_u35 = st.tabs([
            "⚡ Top 20 Over 1.5", "🔒 Top 20 Under 1.5", "🔥 Top 20 Over 2.5", "🛡️ Top 20 Under 2.5", "🚀 Top 20 Over 3.5", "🧱 Top 20 Under 3.5"
        ])
        cols_ou = ["Πρωτάθλημα", "Ημερομηνία", "Ώρα", "Αγώνας", "xG Γηπεδούχου", "xG Φιλοξενούμενου"]
        
        with subtab_o15:
            st.dataframe(df_all.sort_values(by="Over 1.5 %", ascending=False).head(20)[cols_ou + ["Over 1.5 %"]], use_container_width=True)
        with subtab_u15:
            st.dataframe(df_all.sort_values(by="Under 1.5 %", ascending=False).head(20)[cols_ou + ["Under 1.5 %"]], use_container_width=True)
        with subtab_o25:
            st.dataframe(df_all.sort_values(by="Over 2.5 %", ascending=False).head(20)[cols_ou + ["Over 2.5 %"]], use_container_width=True)
        with subtab_u25:
            st.dataframe(df_all.sort_values(by="Under 2.5 %", ascending=False).head(20)[cols_ou + ["Under 2.5 %"]], use_container_width=True)
        with subtab_o35:
            st.dataframe(df_all.sort_values(by="Over 3.5 %", ascending=False).head(20)[cols_ou + ["Over 3.5 %"]], use_container_width=True)
        with subtab_u35:
            st.dataframe(df_all.sort_values(by="Under 3.5 %", ascending=False).head(20)[cols_ou + ["Under 3.5 %"]], use_container_width=True)
    else:
        st.warning("⚠️ Δεν βρέθηκαν επερχόμενοι αγώνες.")

# --- TAB 4: ΑΝΑΛΥΤΙΚΗ ΠΡΟΒΟΛΗ ΠΡΩΤΑΘΛΗΜΑΤΟΣ & AI ---
with tab4:
    st.subheader(f"📊 Αναλυτική Προβολή: {selected_league_name}")
    if selected_league_name in league_predictions and not league_predictions[selected_league_name].empty:
        df_preds = league_predictions[selected_league_name].sort_values(by="Πιθανότητα %", ascending=False)
        top_picks = df_preds[df_preds["Πιθανότητα %"] >= CONFIDENCE_THRESHOLD]
        if not top_picks.empty:
            st.dataframe(top_picks, use_container_width=True)
        else:
            st.warning(f"⚠️ Δεν βρέθηκαν παιχνίδια με πιθανότητα >= {CONFIDENCE_THRESHOLD}%.")
            
        st.divider()
        st.subheader("🤖 AI Tactical Analyst")
        selected_match = st.selectbox("Επιλέξτε αγώνα για AI Ανάλυση:", options=df_preds['Αγώνας'].tolist())
        if st.button("🚀 Εκτέλεση AI Ανάλυσης"):
            match_row = df_preds[df_preds['Αγώνας'] == selected_match].iloc[0]
            h_team, a_team = selected_match.split(" vs ")
            m_date = match_row['Ημερομηνία']
            summary = f"Αγώνας: {selected_match} | xG: {match_row['xG Γηπεδούχου']} - {match_row['xG Φιλοξενούμενου']} | Πιθανότερο Σκορ: {match_row['Πιθανότερο Σκορ']} ({match_row['Πιθανότητα Σκορ %']}%)"
            with st.spinner("🔎 Ανάλυση από το Gemini..."):
                ai_report = get_live_ai_analysis(h_team, a_team, m_date, summary)
            st.info(ai_report)
    else:
        st.warning(f"⚠️ Δεν βρέθηκαν αγώνες για το πρωτάθλημα {selected_league_name}.")
