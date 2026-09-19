import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Football Statistical Predictor", layout="wide")

st.title("📊 Στατιστική Ανάλυση & Προγνωστικά Ποδοσφαίρου")
st.subheader("Μοντέλο Poisson για τα Κύρια Πρωταθλήματα (Αγγλία, Γερμανία, Ισπανία, Ιταλία, Ελλάδα)")

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

# Διορθωμένη γραμμή 24
CONFIDENCE_THRESHOLD = st.sidebar.slider("Ελάχιστο Ποσοστό Σιγουριάς (%)", min_value=50, max_value=85, value=65, step=5)

@st.cache_data(ttl=3600)
def load_data(url):
    try:
        df = pd.read_csv(url)
        # Κρατάμε τις απαραίτητες στήλες: HomeTeam, AwayTeam, FTHG (Full Time Home Goals), FTAG
        df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
        return df
    except Exception as e:
        return None

def calculate_poisson_probs(df):
    """Υπολογίζει τις πιθανότητες Poisson για κάθε ομάδα"""
    avg_home_goals = df['FTHG'].mean()
    avg_away_goals = df['FTAG'].mean()
    
    teams = sorted(list(set(df['HomeTeam']).union(set(df['AwayTeam']))))
    
    # Υπολογισμός επιθετικής & αμυντικής ισχύος
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
    
    # Αναμενόμενα γκολ (xG)
    lambda_home = h_stat['home_attack'] * a_stat['away_defense'] * avg_home_goals
    lambda_away = a_stat['away_attack'] * h_stat['home_defense'] * avg_away_goals
    
    # Πίνακας Πιθανοτήτων Σκορ (0 έως 5 γκολ)
    max_goals = 6
    home_probs = [poisson.pmf(i, lambda_home) for i in range(max_goals)]
    away_probs = [poisson.pmf(i, lambda_away) for i in range(max_goals)]
    
    score_matrix = np.outer(home_probs, away_probs)
    
    # Πιθανότητες 1X2
    prob_home = np.sum(np.tril(score_matrix, -1))
    prob_draw = np.sum(np.diag(score_matrix))
    prob_away = np.sum(np.triu(score_matrix, 1))
    
    # Πιθανότητες Over / Under
    prob_over_1_5 = np.sum(score_matrix[np.add.outer(range(max_goals), range(max_goals)) > 1.5])
    prob_over_2_5 = np.sum(score_matrix[np.add.outer(range(max_goals), range(max_goals)) > 2.5])
    
    # Πιθανότητα BTTS (Goal/Goal)
    prob_btts = np.sum(score_matrix[1:, 1:])
    
    return {
        'xG_Home': round(lambda_home, 2),
        'xG_Away': round(lambda_away, 2),
        'Prob_1': prob_home,
        'Prob_X': prob_draw,
        'Prob_2': prob_away,
        'Prob_Over_1.5': prob_over_1_5,
        'Prob_Over_2.5': prob_over_2_5,
        'Prob_BTTS': prob_btts
    }

df = load_data(league_url)

if df is not None and len(df) > 10:
    stats, avg_h, avg_a = calculate_poisson_probs(df)
    teams = sorted(list(stats.keys()))
    
    st.sidebar.divider()
    st.sidebar.subheader("🔮 Ανάλυση Συγκεκριμένου Αγώνα")
    home_select = st.sidebar.selectbox("Γηπεδούχος", teams, index=0)
    away_select = st.sidebar.selectbox("Φιλοξενούμενος", teams, index=1 if len(teams)>1 else 0)
    
    if home_select != away_select:
        pred = predict_match(home_select, away_select, stats, avg_h, avg_a)
        
        st.divider()
        st.subheader(f"⚔️ {home_select} vs {away_select}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("xG Γηπεδούχου", pred['xG_Home'])
        col2.metric("xG Φιλοξενούμενου", pred['xG_Away'])
        col3.metric("Πιθανότητα 1", f"{pred['Prob_1']*100:.1f}%")
        col4.metric("Πιθανότητα Over 2.5", f"{pred['Prob_Over_2.5']*100:.1f}%")
        
        # Εντοπισμός των πιο «Σίγουρων» Σημείων για τον αγώνα
        tips = []
        if pred['Prob_1'] * 100 >= CONFIDENCE_THRESHOLD:
            tips.append((f"1 (Νίκη {home_select})", pred['Prob_1'] * 100))
        if pred['Prob_2'] * 100 >= CONFIDENCE_THRESHOLD:
            tips.append((f"2 (Νίκη {away_select})", pred['Prob_2'] * 100))
        if (pred['Prob_1'] + pred['Prob_X']) * 100 >= CONFIDENCE_THRESHOLD:
            tips.append(("1X (Διπλή Ευκαιρία)", (pred['Prob_1'] + pred['Prob_X']) * 100))
        if pred['Prob_Over_1.5'] * 100 >= CONFIDENCE_THRESHOLD:
            tips.append(("Over 1.5 Goals", pred['Prob_Over_1.5'] * 100))
        if pred['Prob_Over_2.5'] * 100 >= CONFIDENCE_THRESHOLD:
            tips.append(("Over 2.5 Goals", pred['Prob_Over_2.5'] * 100))
        
        if tips:
            st.success("🎯 **Προτεινόμενα Σημεία Υψηλής Πιθανότητας:**")
            for tip, prob in tips:
                st.write(f"- **{tip}** με στατιστική πιθανότητα **{prob:.1f}%**")
        else:
            st.info("ℹ️ Δεν βρέθηκαν σημεία που να ξεπερνούν το όριο σιγουριάς που θέσατε.")
            
else:
    st.error("⚠️ Δεν ήταν δυνατή η φόρτωση των στατιστικών δεδομένων για το συγκεκριμένο πρωτάθλημα.")
