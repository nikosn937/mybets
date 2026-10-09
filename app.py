import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import math
import re
import time
from google import genai
from google.genai import types

st.set_page_config(page_title="Pro Football Predictor + Cross-League Top Scores", layout="wide")

st.title("⚽ Pro Football Predictor (Dixon-Coles + AI Tactical Analyst)")
st.subheader("Σεζόν 2026/2027 | Στατιστική Ανάλυση 10ετίας, Exponential Decay & Cross-League Top 10 Scores")

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
    """Καθαρίζει τα ονόματα ομάδων για απόλυτη ταύτιση μεταξύ διαφορετικών CSV"""
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
    """Φορτώνει τα δεδομένα της τρέχουσας σεζόν για τον υπολογισμό φόρμας/xG"""
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
    """Φορτώνει αγώνες από τις τελευταίες 10 σεζόν για πλήρη προϊστορία"""
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
        full_df = pd.concat(dfs, ignore_index=True)
        return full_df
    return None

@st.cache_data(ttl=1800)
def load_fixtures_data(code):
    """Φορτώνει το πρόγραμμα επερχόμενων αγώνων και τις αποδόσεις"""
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
    """Υπολογίζει συντελεστή H2H με βάση έως 20 αγώνες της 10ετίας"""
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
    
    home_wins = 0
    away_wins = 0
    draws = 0
    
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
    """Υπολογισμός ισχύος με Εκθετική Απόσβεση Χρόνου (Exponential Time-Decay)"""
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
        
        if h_idx.sum() > 0:
            h_scored = np.average(df.loc[h_idx, 'FTHG'], weights=weights[h_idx])
            h_conceded = np.average(df.loc[h_idx, 'FTAG'], weights=weights[h_idx])
        else:
            h_scored, h_conceded = avg_home_goals, avg_away_goals

        if a_idx.sum() > 0:
            a_scored = np.average(df.loc[a_idx, 'FTAG'], weights=weights[a_idx])
            a_conceded = np.average(df.loc[a_idx, 'FTHG'], weights=weights[a_idx])
        else:
            a_scored, a_conceded = avg_away_goals, avg_home_goals
            
        stats[team] = {
            'home_attack': h_scored / avg_home_goals if avg_home_goals > 0 else 1.0,
            'home_defense': h_conceded / avg_away_goals if avg_away_goals > 0 else 1.0,
            'away_attack': a_scored / avg_away_goals if avg_away_goals > 0 else 1.0,
            'away_defense': a_conceded / avg_home_goals if avg_home_goals > 0 else 1.0
        }
        
    return stats, avg_home_goals, avg_away_goals

def dixon_coles_adjustment(x, y, lambda_h, lambda_a, rho):
    """Διόρθωση Dixon-Coles για χαμηλά σκορ"""
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
    """Υπολογισμός xG, Πιθανοτήτων, BTTS & Top 3 Σκορ"""
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
    
    prob_over_1_5 = np.sum(score_matrix[np.add.outer(range(max_goals), range(max_goals)) > 1.5])
    prob_over_2_5 = np.sum(score_matrix[np.add.outer(range(max_goals), range(max_goals)) > 2.5])
    
    prob_btts_yes = np.sum(score_matrix[1:, 1:])
    prob_btts_no = 1.0 - prob_btts_yes
    
    score_tuples = []
    for x in range(max_goals):
        for y in range(max_goals):
            score_tuples.append((f"{x}-{y}", score_matrix[x, y]))
    score_tuples.sort(key=lambda item: item[1], reverse=True)
    
    top_score_name = score_tuples[0][0]
    top_score_prob = score_tuples[0][1]
    
    top_3_scores = [f"{s[0]} ({round(s[1]*100, 1)}%)" for s in score_tuples[:3]]
    
    return {
        'xG_Home': round(lambda_home, 2),
        'xG_Away': round(lambda_away, 2),
        'Prob_1': prob_home,
        'Prob_X': prob_draw,
        'Prob_2': prob_away,
        'Prob_1X': prob_home + prob_draw,
        'Prob_X2': prob_away + prob_draw,
        'Prob_Over_1.5': prob_over_1_5,
        'Prob_Over_2.5': prob_over_2_5,
        'Prob_GG': prob_btts_yes,
        'Prob_NG': prob_btts_no,
        'Top_Score': top_score_name,
        'Top_Score_Prob': top_score_prob,
        'Top_Scores': ", ".join(top_3_scores)
    }

@st.cache_data(ttl=3600, show_spinner=False)
def get_live_ai_analysis(home_team, away_team, date_str, stats_summary):
    """
    Εκτελεί ανάλυση AI με ανθεκτικότητα σε σφάλματα (Retries & Fallback).
    """
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return "⚠️ Δεν έχει οριστεί το GEMINI_API_KEY στα secrets του Streamlit."

    client = genai.Client(api_key=api_key)

    prompt = f"""
    Είσαι ένας κορυφαίος αθλητικός αναλυτής και ποσοτικός αναλυτής στοιχημάτων (Quantitative Betting Analyst).
    
    Αντικείμενο: Αγώνας {home_team} vs {away_team} στις {date_str}.
    
    Δεδομένα Μοντέλου Dixon-Coles/xG, Exponential Time-Decay & BTTS:
    {stats_summary}
    
    ΑΠΟΣΤΟΛΗ:
    1. Αξιολόγησε τα ποσοτικά δεδομένα (Expected Goals xG, GG/NG, Πιθανότερα Σκορ & Value Bet).
    2. Δώσε μια σύντομη αναφορά (3-4 bullet points) με:
       - 📊 **Τακτική Αξιολόγηση xG & Ισορροπίας**
       - ⚽ **Εκτίμηση BTTS (GG/NG) & Πιθανότερου Σκορ**
       - 🎯 **Τελικό AI Verdict & Διαχείριση Ρίσκου** (Επιβεβαίωση σημείου, Reroute σε 1X/X2/GG, ή Αποχή).
    
    Γράψε την απάντηση στα Ελληνικά, σύντομα και επαγγελματικά.
    """

    models_to_try = ['gemini-3.6-flash', 'gemini-1.5-flash']

    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                err_msg = str(e)
                if "503" in err_msg or "UNAVAILABLE" in err_msg:
                    time.sleep(2 * (attempt + 1))
                    continue
                elif "404" in err_msg or "NOT_FOUND" in err_msg:
                    break
                else:
                    return f"❌ Σφάλμα κατά τη λειτουργία AI: {e}"

    return "⚠️ Οι διακομιστές της Google είναι προσωρινά μη διαθέσιμοι. Παρακαλώ δοκιμάστε ξανά σε λίγα δευτερόλεπτα."

# --- ΦΙΛΤΡΟ ΗΜΕΡΟΜΗΝΙΩΝ ΣΤΗ SIDEBAR ---
st.sidebar.divider()
st.sidebar.subheader("📅 Φίλτρο Ημερομηνιών")

# Φόρτωση πρώτης λίγκας για αρχικές ημερομηνίες
init_fixtures = load_fixtures_data(LEAGUES[selected_league_name]["code"])
min_f_date = init_fixtures['Date'].min().date() if init_fixtures is not None and not init_fixtures.empty else pd.Timestamp.today().date()
max_f_date = init_fixtures['Date'].max().date() if init_fixtures is not None and not init_fixtures.empty else pd.Timestamp.today().date()

date_range = st.sidebar.date_input(
    "Επιλέξτε εύρος ημερομηνιών (για όλα τα πρωταθλήματα)",
    value=(min_f_date, max_f_date)
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
elif isinstance(date_range, tuple) and len(date_range) == 1:
    start_date, end_date = date_range[0], max_f_date
else:
    start_date, end_date = min_f_date, max_f_date

# --- ΣΑΡΩΣΗ ΟΛΩΝ ΤΩΝ ΠΡΩΤΑΘΛΗΜΑΤΩΝ ΓΙΑ ΤΑ TOP 10 ΣΚΟΡ ---
all_league_scores = []
league_predictions = {}

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
            
            if USE_H2H:
                h2h_h_mult, h2h_a_mult, h2h_str = calculate_h2h_adjustment(df_h2h, h_team, a_team, max_matches=20)
            else:
                h2h_h_mult, h2h_a_mult, h2h_str = 1.0, 1.0, "Off"
                
            pred = predict_match_dc(h_team, a_team, l_stats, l_avg_h, l_avg_a, RHO_DIXON, 1.0, 1.0, h2h_h_mult, h2h_a_mult)
            
            outcomes = [
                ("1", pred['Prob_1']),
                ("2", pred['Prob_2']),
                ("1X", pred['Prob_1X']),
                ("X2", pred['Prob_X2']),
                ("Over 1.5", pred['Prob_Over_1.5']),
                ("Over 2.5", pred['Prob_Over_2.5']),
                ("GG (Goal/Goal)", pred['Prob_GG']),
                ("NG (No Goal)", pred['Prob_NG'])
            ]
            
            best_pick, best_prob = max(outcomes, key=lambda x: x[1])
            
            value_flag = "—"
            if 'B365H' in row and pd.notna(row['B365H']):
                odd_h = row['B365H']
                odd_a = row['B365A']
                if best_pick == "1" and (pred['Prob_1'] > (1 / odd_h)):
                    value_flag = f"🔥 Value 1 (@{odd_h})"
                elif best_pick == "2" and (pred['Prob_2'] > (1 / odd_a)):
                    value_flag = f"🔥 Value 2 (@{odd_a})"
            
            match_data = {
                "Πρωτάθλημα": l_name,
                "Ημερομηνία": m_date,
                "Ώρα": m_time,
                "Αγώνας": f"{h_team} vs {a_team}",
                "Προτεινόμενο Σημείο": best_pick,
                "Πιθανότητα %": round(best_prob * 100, 1),
                "GG %": round(pred['Prob_GG'] * 100, 1),
                "Πιθανότερο Σκορ": pred['Top_Score'],
                "Πιθανότητα Σκορ %": round(pred['Top_Score_Prob'] * 100, 1),
                "Πιθανότερα Σκορ": pred['Top_Scores'],
                "Value Bet": value_flag,
                "Προϊστορία (H2H 10ετίας)": h2h_str,
                "xG Γηπεδούχου": pred['xG_Home'],
                "xG Φιλοξενούμενου": pred['xG_Away']
            }
            
            preds_list.append(match_data)
            all_league_scores.append(match_data)
            
        league_predictions[l_name] = pd.DataFrame(preds_list)

# --- ΕΜΦΑΝΙΣΗ 10 ΠΙΟ ΠΙΘΑΝΩΝ ΣΚΟΡ ΑΠΟ ΟΛΑ ΤΑ ΠΡΩΤΑΘΛΗΜΑΤΑ ---
st.info(f"📅 **Περίοδος Αγώνων: {start_date.strftime('%d/%m/%Y')} έως {end_date.strftime('%d/%m/%Y')}**")

if all_league_scores:
    df_all_scores = pd.DataFrame(all_league_scores)
    df_top_10_scores = df_all_scores.sort_values(by="Πιθανότητα Σκορ %", ascending=False).head(10)
    
    st.subheader("🎯 Top 10 Πιο Πιθανά Σκορ (Όλα τα Πρωτάθληματα)")
    st.caption("Τα 10 ακριβή σκορ με τις υψηλότερες πιθανότητες επαλήθευσης από όλους τους αγώνες του επιλεγμένου διαστήματος.")
    
    display_cols_scores = [
        "Πρωτάθλημα", "Ημερομηνία", "Ώρα", "Αγώνας", 
        "Πιθανότερο Σκορ", "Πιθανότητα Σκορ %", "xG Γηπεδούχου", "xG Φιλοξενούμενου"
    ]
    st.dataframe(df_top_10_scores[display_cols_scores], use_container_width=True)
    st.divider()

# --- ΑΝΑΛΥΤΙΚΗ ΠΡΟΒΟΛΗ ΕΠΙΛΕΓΜΕΝΟΥ ΠΡΩΤΑΘΛΗΜΑΤΟΣ ---
st.subheader(f"📊 Αναλυτική Προβολή: {selected_league_name}")

if selected_league_name in league_predictions and not league_predictions[selected_league_name].empty:
    df_preds = league_predictions[selected_league_name].sort_values(by="Πιθανότητα %", ascending=False)
    
    st.subheader("🔥 Top Σημεία (Dixon-Coles + Time-Decay, GG/NG & Value Bets)")
    top_picks = df_preds[df_preds["Πιθανότητα %"] >= CONFIDENCE_THRESHOLD]
    
    if not top_picks.empty:
        st.dataframe(top_picks, use_container_width=True)
    else:
        st.warning(f"⚠️ Δεν βρέθηκαν επερχόμενα παιχνίδια με πιθανότητα >= {CONFIDENCE_THRESHOLD}% στο επιλεγμένο διάστημα.")
        
    st.divider()
    
    with st.expander("📊 Προβολή Όλων των Αναλυμένων Αγώνων Πρωταθλήματος"):
        st.dataframe(df_preds, use_container_width=True)

    # --- ΕΝΟΤΗΤΑ AI ANALYST ---
    st.divider()
    st.subheader("🤖 AI Tactical & Quantitative Analyst")
    st.caption("Επιλέξτε έναν αγώνα για να εκτελέσει το Gemini 3.6 Flash ποσοτική και τακτική ανάλυση.")
    
    selected_match = st.selectbox(
        "Επιλέξτε αγώνα για AI Ανάλυση:",
        options=df_preds['Αγώνας'].tolist()
    )
    
    if st.button("🚀 Εκτέλεση AI Ανάλυσης"):
        match_row = df_preds[df_preds['Αγώνας'] == selected_match].iloc[0]
        h_team, a_team = selected_match.split(" vs ")
        m_date = match_row['Ημερομηνία']
        
        summary = f"""
        - Πρωτάθλημα: {match_row['Πρωτάθλημα']}
        - Προτεινόμενο Σημείο Μοντέλου: {match_row['Προτεινόμενο Σημείο']} ({match_row['Πιθανότητα %']}%)
        - xG: {match_row['xG Γηπεδούχου']} - {match_row['xG Φιλοξενούμενου']}
        - Πιθανότητα GG (Goal/Goal): {match_row['GG %']}%
        - Πιθανότερο Σκορ: {match_row['Πιθανότερο Σκορ']} ({match_row['Πιθανότητα Σκορ %']}%)
        - Πιθανότερα Σκορ: {match_row['Πιθανότερα Σκορ']}
        - Προϊστορία H2H (10ετίας): {match_row['Προϊστορία (H2H 10ετίας)']}
        - Value Bet: {match_row['Value Bet']}
        """
        
        with st.spinner("🔎 Πραγματοποιείται ανάλυση από το Gemini..."):
            ai_report = get_live_ai_analysis(h_team, a_team, m_date, summary)
            
        st.markdown("### 🤖 AI Report & Verdict")
        st.info(ai_report)

else:
    st.warning(f"⚠️ Δεν βρέθηκαν επερχόμενοι αγώνες για το πρωτάθλημα {selected_league_name} στο συγκεκριμένο εύρος ημερομηνιών.")
```[cite: 3]

---

### 🔥 Τι άλλαξε & Πώς Λειτουργεί:

1. **Δια-πρωταθληματική Σάρωση (Cross-League Scan)**: Η εφαρμογή υπολογίζει τώρα αυτόματα όλα τα παιχνίδια από **όλα τα διαθέσιμα πρωταθλήματα** (Premier League, La Liga, Bundesliga, Serie A, Super League) για τις ημερομηνίες που έχετε επιλέξει[cite: 3].
2. **Νέος Πίνακας `Top 10 Πιο Πιθανά Σκορ`**: Στην κορυφή της σελίδας εμφανίζεται ένας νέος πίνακας που απομονώνει και ταξινομεί τα **10 πιο πιθανά ακριβή σκορ** από ολόκληρο το πρόγραμμα της περιόδου, δείχνοντας το Πρωτάθλημα, τον Αγώνα, το Πιθανότερο Σκορ (π.χ. `1-1`, `2-0`, `1-0`), την Πιθανότητά του %, καθώς και τα xG των ομάδων[cite: 3].
3. **Διατήρηση Αναλυτικού Πρωταθλήματος**: Κάτω από τον πίνακα των Top 10 σκορ, διατηρείται η αναλυτική προβολή για το πρωτάθλημα που επιλέγετε από τη Sidebar μαζί με τον AI Analyst[cite: 3].
