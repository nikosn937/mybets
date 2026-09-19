import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Football Statistical Predictor", layout="wide")

st.title("📊 Στατιστική Ανάλυση & Top 5 Σίγουρα Σημεία")
st.subheader("Μοντέλο Poisson & Φιλτράρισμα Αγώνων ανά Ημερομηνία")

# Επιλογή Πρωταθλήματος
LEAGUES = {
    "Premier League (Αγγλία)": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "La Liga (Ισπανία)": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "Bundesliga (Γερμανία)": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "Serie A (Ιταλία)": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "Super League (Ελλάδα)": "https://www.football-data.co.uk/mmz4281/2526/G1.csv"
}

st.sidebar.header("⚙️ Παράμετροι Ανάλυσης")
selected_league_name = st.sidebar.selectbox("Επιλέξτε Πρωτάθλημα", list(LEAGUES.keys()))
league_url = LEAGUES[selected_league_name]

CONFIDENCE_THRESHOLD = st.sidebar.slider("Ελάχιστο Ποσοστό Σιγουριάς (%)", min_value=50, max_value=90, value=65, step=5)

@st.cache_data(ttl=3600)
def load_data(url):
    try:
        df = pd.read_csv(url)
        # Κρατάμε τις απαραίτητες στήλες
        df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
        # Μετατροπή ημερομηνίας σε datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['Date'])
        return df
    except Exception as e:
        return None

def calculate_poisson_probs(df):
    """Υπολογίζει τις πιθανότητες Poisson για κάθε ομάδα"""
    avg_home_goals = df['FTHG'].mean()
    avg_away_goals = df['FTAG'].mean()
    
    teams = sorted(list(set(df['HomeTeam']).union(set(df['AwayTeam']))))
    
    stats = {}
    for team in teams:
        home_games = df[df['HomeTeam'] == team]
        away_games = df[df['AwayTeam'] == team]
        
        home_scored = home_games['FTHG'].mean() if len(home_games) > 0 else avg_home_goals
        home_conceded = home_games['FTAG'].mean() if len(home_games) > 0 else avg_away_goals
        
        away_scored = away_games['FTAG'].mean() if len(away_games) > 0 else avg_away_goals
        away_conceded = away_games['FTHG'].mean() if len(away_games) > 0 else avg_home_goals
        
        stats[team] = {
            'home_attack': home_scored / avg_home_goals if avg_home_goals > 0 else 1,
            'home_defense': home_conceded / avg_away_goals if avg_away_goals > 0 else 1,
            'away_attack': away_scored / avg_away_goals if avg_away_goals > 0 else 1,
            'away_defense': away_conceded / avg_home_goals if avg_home_goals > 0 else 1
        }
    
    return stats, avg_home_goals, avg_away_goals

def predict_match(home_team, away_team, stats, avg_home_goals, avg_away_goals):
    """Υπολογίζει τα πιθανά σκορ και τις πιθανότητες αγορών"""
    h_stat = stats[home_team]
    a_stat = stats[away_team]
    
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

df = load_data(league_url)

if df is not None and len(df) > 10:
    stats, avg_h, avg_a = calculate_poisson_probs(df)
    
    # Φιλτράρισμα Ημερομηνιών στη Sidebar
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    st.sidebar.divider()
    st.sidebar.subheader("📅 Φίλτρο Ημερομηνιών")
    date_range = st.sidebar.date_input(
        "Επιλέξτε εύρος ημερομηνιών",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
    else:
        filtered_df = df
        
    st.write(f"🔍 **Εμφάνιση αγώνων από {start_date.strftime('%d/%m/%Y')} έως {end_date.strftime('%d/%m/%Y')}** ({len(filtered_df)} αγώνες)")

    # Υπολογισμός σημείων για όλους τους φιλτραρισμένους αγώνες
    all_predictions = []
    
    for idx, row in filtered_df.iterrows():
        h_team = row['HomeTeam']
        a_team = row['AwayTeam']
        match_date = row['Date'].strftime('%d/%m/%Y')
        
        pred = predict_match(h_team, a_team, stats, avg_h, avg_a)
        
        # Λίστα πιθανών σημείων ανά αγώνα
        outcomes = [
            ("1", pred['Prob_1']),
            ("2", pred['Prob_2']),
            ("1X", pred['Prob_1X']),
            ("X2", pred['Prob_X2']),
            ("Over 1.5", pred['Prob_Over_1.5']),
            ("Over 2.5", pred['Prob_Over_2.5'])
        ]
        
        # Εντοπισμός του σημείου με την υψηλότερη πιθανότητα για τον συγκεκριμένο αγώνα
        best_pick, best_prob = max(outcomes, key=lambda x: x[1])
        
        all_predictions.append({
            "Ημερομηνία": match_date,
            "Αγώνας": f"{h_team} vs {a_team}",
            "Προτεινόμενο Σημείο": best_pick,
            "Πιθανότητα %": round(best_prob * 100, 1),
            "xG Γηπεδούχου": pred['xG_Home'],
            "xG Φιλοξενούμενου": pred['xG_Away']
        })
        
    df_preds = pd.DataFrame(all_predictions)
    
    # Ταξινόμηση βάσει Πιθανότητας % (Descending)
    df_preds = df_preds.sort_values(by="Πιθανότητα %", ascending=False)
    
    # 🌟 ΕΜΦΑΝΙΣΗ TOP 5 ΣΙΓΟΥΡΩΝ ΣΗΜΕΙΩΝ
    st.subheader("🔥 Top 5 «Σίγουρα» Σημεία της Επιλεγμένης Περιόδου")
    top_5 = df_preds[df_preds["Πιθανότητα %"] >= CONFIDENCE_THRESHOLD].head(5)
    
    if not top_5.empty:
        st.dataframe(top_5, use_container_width=True)
    else:
        st.info(f"ℹ️ Δεν βρέθηκαν σημεία με πιθανότητα >= {CONFIDENCE_THRESHOLD}% στο επιλεγμένο εύρος ημερομηνιών.")
        
    st.divider()
    
    # Πλήρης Πίνακας Αγώνων
    with st.expander("📊 Προβολή Όλων των Αναλυμένων Αγώνων"):
        st.dataframe(df_preds, use_container_width=True)

else:
    st.error("⚠️ Δεν ήταν δυνατή η φόρτωση των στατιστικών δεδομένων.")
