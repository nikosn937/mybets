import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Football Statistical Predictor 2026/27", layout="wide")

st.title("📊 Προγνωστικά Επερχόμενων Αγώνων (Σεζόν 2026/2027)")
st.subheader("Στατιστική Ανάλυση Poisson & Φόρμα Ομάδων")

# Ορισμός Σεζόν 2026/2027 (Κωδικός 2627)
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

st.sidebar.header("⚙️ Παράμετροι Ανάλυσης")
selected_league_name = st.sidebar.selectbox("Επιλέξτε Πρωτάθλημα", list(LEAGUES.keys()))
history_url = LEAGUES[selected_league_name]["history"]
league_code = LEAGUES[selected_league_name]["code"]

CONFIDENCE_THRESHOLD = st.sidebar.slider("Ελάχιστο Ποσοστό Σιγουριάς (%)", min_value=50, max_value=90, value=65, step=5)
RECENT_WEIGHT = st.sidebar.checkbox("Δώσε μεγαλύτερη βαρύτητα στα πρόσφατα παιχνίδια (Φόρμα)", value=True)

@st.cache_data(ttl=1800)
def load_history_data(url):
    """Φορτώνει τα παιχνίδια της σεζόν 2026/2027"""
    try:
        df = pd.read_csv(url)
        df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        return df.dropna(subset=['Date']).sort_values('Date')
    except Exception:
        return None

@st.cache_data(ttl=1800)
def load_fixtures_data(code):
    """Φορτώνει το πρόγραμμα των επερχόμενων αγώνων"""
    try:
        url = "https://www.football-data.co.uk/fixtures.csv"
        df = pd.read_csv(url)
        df = df[df['Div'] == code]
        df = df[['Date', 'Time', 'HomeTeam', 'AwayTeam']].dropna(subset=['HomeTeam', 'AwayTeam'])
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        return df.dropna(subset=['Date'])
    except Exception:
        return None

def calculate_weighted_poisson(df, use_weights=True):
    """Υπολογίζει τις δυνάμεις των ομάδων με βάση τη σεζόν 2026/2027"""
    if use_weights and len(df) > 1:
        n = len(df)
        weights = np.linspace(0.5, 1.5, n)
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
            'home_attack': h_scored / avg_home_goals if avg_home_goals > 0 else 1,
            'home_defense': h_conceded / avg_away_goals if avg_away_goals > 0 else 1,
            'away_attack': a_scored / avg_away_goals if avg_away_goals > 0 else 1,
            'away_defense': a_conceded / avg_home_goals if avg_home_goals > 0 else 1
        }
        
    return stats, avg_home_goals, avg_away_goals

def predict_match(home_team, away_team, stats, avg_home_goals, avg_away_goals):
    """Υπολογίζει πιθανότητες για έναν μελλοντικό αγώνα"""
    default_stat = {'home_attack': 1, 'home_defense': 1, 'away_attack': 1, 'away_defense': 1}
    h_stat = stats.get(home_team, default_stat)
    a_stat = stats.get(away_team, default_stat)
    
    lambda_home = h_stat['home_attack'] * a_stat['away_defense'] * avg_home_goals
    lambda_away = a_stat['away_attack'] * h_stat['home_defense'] * avg_away_goals
    
    max_goals = 6
    home_probs = [poisson.pmf(i, lambda_home) for i in range(max_goals)]
    away_probs = [poisson.pmf(i, lambda_away) for i in range(max_goals)]
    
    score_matrix = np.outer(home_probs, away_probs)
    
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

# Φόρτωση Δεδομένων
df_history = load_history_data(history_url)
df_fixtures = load_fixtures_data(league_code)

if df_history is not None and not df_history.empty and df_fixtures is not None and not df_fixtures.empty:
    
    # Υπολογισμός στατιστικών με βάση τους αγώνες της σεζόν 2026/2027
    stats, avg_h, avg_a = calculate_weighted_poisson(df_history, RECENT_WEIGHT)
    
    st.success(f"✅ Φορτώθηκαν **{len(df_history)} διεξαχθέντες αγώνες** της σεζόν **2026/2027**.")
    
    # Φιλτράρισμα Ημερομηνιών Επερχόμενων Αγώνων
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
        
        pred = predict_match(h_team, a_team, stats, avg_h, avg_a)
        
        outcomes = [
            ("1", pred['Prob_1']),
            ("2", pred['Prob_2']),
            ("1X", pred['Prob_1X']),
            ("X2", pred['Prob_X2']),
            ("Over 1.5", pred['Prob_Over_1.5']),
            ("Over 2.5", pred['Prob_Over_2.5'])
        ]
        
        best_pick, best_prob = max(outcomes, key=lambda x: x[1])
        
        all_predictions.append({
            "Ημερομηνία": match_date,
            "Ώρα": match_time,
            "Αγώνας": f"{h_team} vs {a_team}",
            "Προτεινόμενο Σημείο": best_pick,
            "Πιθανότητα %": round(best_prob * 100, 1),
            "xG Γηπεδούχου": pred['xG_Home'],
            "xG Φιλοξενούμενου": pred['xG_Away']
        })
        
    df_preds = pd.DataFrame(all_predictions)
    
    if not df_preds.empty:
        df_preds = df_preds.sort_values(by="Πιθανότητα %", ascending=False)
        
        st.subheader("🔥 Top Σίγουρα Σημεία για τους Επόμενους Αγώνες")
        top_picks = df_preds[df_preds["Πιθανότητα %"] >= CONFIDENCE_THRESHOLD]
        
        if not top_picks.empty:
            st.dataframe(top_picks, use_container_width=True)
        else:
            st.warning(f"⚠️ Δεν βρέθηκαν επερχόμενα παιχνίδια με πιθανότητα >= {CONFIDENCE_THRESHOLD}% στο επιλεγμένο διάστημα.")
            
        st.divider()
        
        with st.expander("📊 Προβολή Όλων των Αναλυμένων Επερχόμενων Αγώνων"):
            st.dataframe(df_preds, use_container_width=True)
    else:
        st.warning("⚠️ Δεν βρέθηκαν επερχόμενοι αγώνες στο συγκεκριμένο εύρος ημερομηνιών.")

elif df_fixtures is None or df_fixtures.empty:
    st.warning("ℹ️ Δεν υπάρχουν διαθέσιμοι επερχόμενοι αγώνες στο πρόγραμμα αυτή τη στιγμή για το συγκεκριμένο πρωτάθλημα.")
else:
    st.error("⚠️ Σφάλμα κατά τη φόρτωση των στατιστικών δεδομένων της σεζόν 2026/2027.")
