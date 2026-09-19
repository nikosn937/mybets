import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import math
import re
from google import genai
from google.genai import types

st.set_page_config(page_title="Pro Football Predictor + AI Live Scanner", layout="wide")

st.title("⚽ Pro Football Predictor (Dixon-Coles + Dynamic AI Model Scanner)")
st.subheader("Σεζόν 2026/2027 | Στατιστική Ανάλυση & Live AI News/Weather Scanner")

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
selected_league_name = st.sidebar.selectbox("Επιλέξτε Πρωτάθλημα", list(LEAGUES.keys()))
history_url = LEAGUES[selected_league_name]["history"]
league_code = LEAGUES[selected_league_name]["code"]

CONFIDENCE_THRESHOLD = st.sidebar.slider("Ελάχιστο Ποσοστό Σιγουριάς (%)", min_value=50, max_value=90, value=60, step=5)
USE_WEIGHTS = st.sidebar.checkbox("Στάθμιση Πρόσφατης Φόρμας (Time-Decay)", value=True)
USE_H2H = st.sidebar.checkbox("Ενεργοποίηση Προσαρμογής H2H (Προϊστορία)", value=True)
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
    """Φορτώνει αγώνες από τις 3 τελευταίες σεζόν για να βρει πλούσια προϊστορία"""
    seasons = ["2627", "2526", "2425"]
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

def calculate_h2h_adjustment(df_h2h_all, home_team, away_team, max_matches=6):
    """Υπολογίζει συντελεστή H2H με βάση πολυετή δεδομένα"""
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

def calculate_dixon_coles_stats(df, use_weights=True):
    """Υπολογισμός Εντός/Εκτός επιθετικής & αμυντικής ισχύος ομάδων"""
    if use_weights and len(df) > 1:
        n = len(df)
        weights = np.linspace(0.4, 1.6, n)
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
    """Υπολογισμός xG και πιθανοτήτων αγώνα"""
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
    
    return {
        'xG_Home': round(lambda_home, 2),
        'xG_Away': round(lambda_away, 2),
        'Prob_1': prob_home,
        'Prob_X': prob_draw,
        'Prob_2': prob_away,
        'Prob_1X': prob_home + prob_draw,
        'Prob_X2': prob_away + prob_draw,
        'Prob_Over_1.5': prob_over_1_5,
        'Prob_Over_2.5': prob_over_2_5
    }

def find_best_available_model(client):
    """
    Ψάχνει αυτόματα στη σουίτα Gemini ποια μοντέλα είναι διαθέσιμα 
    και επιστρέφει το καταλληλότερο μοντέλο Flash.
    """
    try:
        models = client.models.list()
        supported_models = []
        
        for m in models:
            # Έλεγχος αν υποστηρίζει παραγωγή περιεχομένου
            methods = getattr(m, 'supported_generation_methods', []) or []
            if 'generateContent' in methods or not methods:
                supported_models.append(m.name)
        
        # Προτίμηση στα νεότερα flash μοντέλα με σειρά προτεραιότητας
        for m_name in supported_models:
            clean_name = m_name.replace("models/", "")
            if "flash" in clean_name.lower():
                return clean_name
                
        # Αν δεν βρεθεί flash, επιστροφή του πρώτου διαθέσιμου μοντέλου
        if supported_models:
            return supported_models[0].replace("models/", "")
            
    except Exception:
        pass
        
    # Default fallback
    return "gemini-2.5-flash"

def get_live_ai_analysis(home_team, away_team, date_str, stats_summary):
    """
    Εντοπίζει αυτόματα το ενεργό μοντέλο Gemini και εκτελεί live αναζήτηση 
    στο διαδίκτυο με Google Search Grounding.
    """
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return "⚠️ Δεν έχει οριστεί το GEMINI_API_KEY στα secrets του Streamlit."

    client = genai.Client(api_key=api_key)

    # Αυτόματος εντοπισμός του σωστού μοντέλου
    active_model = find_best_available_model(client)

    prompt = f"""
    Είσαι ένας κορυφαίος αθλητικός αναλυτής και ποσοτικός αναλυτής στοιχημάτων (Quantitative Betting Analyst).
    
    Αντικείμενο: Αγώνας {home_team} vs {away_team} στις {date_str}.
    
    Δεδομένα Μοντέλου Dixon-Coles/xG:
    {stats_summary}
    
    ΑΠΟΣΤΟΛΗ:
    1. Κάνε αναζήτηση στο διαδίκτυο για τα τελευταία νέα (last minute news), τραυματισμούς, τιμωρίες, πιθανές ενδεκάδες και τις καιρικές συνθήκες για τον αγώνα {home_team} vs {away_team}.
    2. Αξιολόγησε αν τα νέα της τελευταίας στιγμής επιβεβαιώνουν ή αμφισβητούν την πρόβλεψη του στατιστικού μοντέλου.
    3. Δώσε μια σύντομη αναφορά (3-4 bullet points) με:
       - 🚑 **Σημαντικές Απουσίες / Ειδήσεις**
       - 🌧️ **Καιρικές Συνθήκες & Γήπεδο**
       - 🎯 **Τελικό AI Verdict & Ρίσκο** (π.χ. Επιβεβαίωση σημείου, Reroute σε 1X, ή Αποχή λόγω υψηλού ρίσκου).
    
    Γράψε την απάντηση στα Ελληνικά, σύντομα και επαγγελματικά.
    """

    try:
        st.toast(f"🤖 Χρήση μοντέλου: `{active_model}`")
        response = client.models.generate_content(
            model=active_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        return response.text, active_model
    except Exception as e:
        return f"❌ Σφάλμα κατά τη live ανάλυση AI ({active_model}): {e}", active_model

# --- ΚΥΡΙΩΣ ΡΟΗ ΕΦΑΡΜΟΓΗΣ ---

df_history = load_history_data(history_url)
df_h2h_all = load_multi_season_h2h(league_code)
df_fixtures = load_fixtures_data(league_code)

if df_history is not None and not df_history.empty and df_fixtures is not None and not df_fixtures.empty:
    
    stats, avg_h, avg_a = calculate_dixon_coles_stats(df_history, USE_WEIGHTS)
    
    st.sidebar.divider()
    st.sidebar.subheader("🚑 Διαχείριση Απουσιών / Φόρμας")
    all_teams = sorted(list(stats.keys()))
    
    selected_adj_team = st.sidebar.selectbox("Επιλέξτε ομάδα για προσαρμογή", ["Καμία"] + all_teams)
    home_adj_factor = 1.0
    
    if selected_adj_team != "Καμία":
        mod_type = st.sidebar.radio("Προσαρμογή λόγω απουσιών:", ["Πλήρης Ομάδα (100%)", "Mικρή Απουσία (-10%)", "Σημαντικές Απουσίες (-20%)"])
        if "Mικρή" in mod_type:
            home_adj_factor = 0.90
        elif "Σημαντικές" in mod_type:
            home_adj_factor = 0.80

    min_f_date = df_fixtures['Date'].min().date()
    max_f_date = df_fixtures['Date'].max().date()
    
    st.sidebar.divider()
    st.sidebar.subheader("📅 Φίλτρο Επερχόμενων Αγώνων")
    date_range = st.sidebar.date_input(
        "Επιλέξτε εύρος ημερομηνιών",
        value=(min_f_date, max_f_date),
        min_value=min_f_date,
        max_value=max_f_date
    )
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_fixtures = df_fixtures[(df_fixtures['Date'].dt.date >= start_date) & (df_fixtures['Date'].dt.date <= end_date)]
    else:
        start_date, end_date = min_f_date, max_f_date
        filtered_fixtures = df_fixtures
        
    st.info(f"📅 **Εμφάνιση επερχόμενων αγώνων από {start_date.strftime('%d/%m/%Y')} έως {end_date.strftime('%d/%m/%Y')}** ({len(filtered_fixtures)} αγώνες)")

    all_predictions = []
    
    for idx, row in filtered_fixtures.iterrows():
        h_team = row['HomeTeam']
        a_team = row['AwayTeam']
        match_date = row['Date'].strftime('%d/%m/%Y')
        match_time = row['Time'] if 'Time' in row and pd.notna(row['Time']) else ""
        
        # Υπολογισμός H2H προσαρμογής
        if USE_H2H:
            h2h_h_mult, h2h_a_mult, h2h_str = calculate_h2h_adjustment(df_h2h_all, h_team, a_team)
        else:
            h2h_h_mult, h2h_a_mult, h2h_str = 1.0, 1.0, "Off"
            
        # Εφαρμογή προσαρμογής απουσιών
        h_adj = home_adj_factor if h_team == selected_adj_team else 1.0
        a_adj = home_adj_factor if a_team == selected_adj_team else 1.0
        
        # Πρόβλεψη Αγώνα
        pred = predict_match_dc(h_team, a_team, stats, avg_h, avg_a, RHO_DIXON, h_adj, a_adj, h2h_h_mult, h2h_a_mult)
        
        outcomes = [
            ("1", pred['Prob_1']),
            ("2", pred['Prob_2']),
            ("1X", pred['Prob_1X']),
            ("X2", pred['Prob_X2']),
            ("Over 1.5", pred['Prob_Over_1.5']),
            ("Over 2.5", pred['Prob_Over_2.5'])
        ]
        
        best_pick, best_prob = max(outcomes, key=lambda x: x[1])
        
        # Υπολογισμός Value Bet
        value_flag = "—"
        if 'B365H' in row and pd.notna(row['B365H']):
            odd_h = row['B365H']
            odd_a = row['B365A']
            if best_pick == "1" and (pred['Prob_1'] > (1 / odd_h)):
                value_flag = f"🔥 Value 1 (@{odd_h})"
            elif best_pick == "2" and (pred['Prob_2'] > (1 / odd_a)):
                value_flag = f"🔥 Value 2 (@{odd_a})"
        
        all_predictions.append({
            "Ημερομηνία": match_date,
            "Ώρα": match_time,
            "Αγώνας": f"{h_team} vs {a_team}",
            "Προτεινόμενο Σημείο": best_pick,
            "Πιθανότητα %": round(best_prob * 100, 1),
            "Value Bet": value_flag,
            "Προϊστορία (H2H)": h2h_str,
            "xG Γηπεδούχου": pred['xG_Home'],
            "xG Φιλοξενούμενου": pred['xG_Away']
        })
        
    df_preds = pd.DataFrame(all_predictions)
    
    if not df_preds.empty:
        df_preds = df_preds.sort_values(by="Πιθανότητα %", ascending=False)
        
        st.subheader("🔥 Top Σημεία (Dixon-Coles, H2H & Value Bets)")
        top_picks = df_preds[df_preds["Πιθανότητα %"] >= CONFIDENCE_THRESHOLD]
        
        if not top_picks.empty:
            st.dataframe(top_picks, use_container_width=True)
        else:
            st.warning(f"⚠️ Δεν βρέθηκαν επερχόμενα παιχνίδια με πιθανότητα >= {CONFIDENCE_THRESHOLD}% στο επιλεγμένο διάστημα.")
            
        st.divider()
        
        with st.expander("📊 Προβολή Όλων των Αναλυμένων Αγώνων"):
            st.dataframe(df_preds, use_container_width=True)

        # --- ΕΝΟΤΗΤΑ LIVE AI ANALYST ---
        st.divider()
        st.subheader("🔍 Live AI Tactical & Last-Minute News Scanner")
        st.caption("Επιλέξτε έναν αγώνα για αυτόματο εντοπισμό διαθέσιμου μοντέλου Gemini και live αναζήτηση στο διαδίκτυο.")
        
        selected_match = st.selectbox(
            "Επιλέξτε αγώνα για Live AI Ανάλυση:",
            options=df_preds['Αγώνας'].tolist()
        )
        
        if st.button("🚀 Εκτέλεση Live AI Scanner"):
            match_row = df_preds[df_preds['Αγώνας'] == selected_match].iloc[0]
            h_team, a_team = selected_match.split(" vs ")
            m_date = match_row['Ημερομηνία']
            
            summary = f"""
            - Προτεινόμενο Σημείο Μοντέλου: {match_row['Προτεινόμενο Σημείο']} ({match_row['Πιθανότητα %']}%)
            - xG: {match_row['xG Γηπεδούχου']} - {match_row['xG Φιλοξενούμενου']}
            - Προϊστορία H2H: {match_row['Προϊστορία (H2H)']}
            - Value Bet: {match_row['Value Bet']}
            """
            
            with st.spinner("🔎 Αναζήτηση ενεργού μοντέλου Gemini & εκτέλεση live αναζήτησης..."):
                ai_report, used_model = get_live_ai_analysis(h_team, a_team, m_date, summary)
                
            st.markdown(f"### 🤖 Live AI Report & Verdict (Μοντέλο: `{used_model}`)")
            st.info(ai_report)

    else:
        st.warning("⚠️ Δεν βρέθηκαν επερχόμενοι αγώνες στο συγκεκριμένο εύρος ημερομηνιών.")

elif df_fixtures is None or df_fixtures.empty:
    st.warning("ℹ️ Δεν υπάρχουν διαθέσιμοι επερχόμενοι αγώνες στο πρόγραμμα αυτή τη στιγμή για το συγκεκριμένο πρωτάθλημα.")
else:
    st.error("⚠️ Σφάλμα κατά τη φόρτωση των στατιστικών δεδομένων.")
