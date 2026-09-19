import streamlit as st
import requests
import pandas as pd

# Ρύθμιση σελίδας
st.set_page_config(page_title="Σύγκριση Αποδόσεων: Bet365 vs Stoiximan", layout="wide")

st.title("⚽ Σύγκριση Αποδόσεων Ποδοσφαίρου")
st.subheader("Bet365 vs Stoiximan (Cyprus / EU)")

# 1. Αυτόματη ανάκτηση του API Key από τα Streamlit Secrets
if "ODDS_API_KEY" in st.secrets:
    API_KEY = st.secrets["ODDS_API_KEY"]
else:
    st.error("⚠️ Δεν βρέθηκε το 'ODDS_API_KEY' στα Secrets. Προσθέστε το στο secrets.toml ή στις ρυθμίσεις του Streamlit Cloud.")
    API_KEY = None

# Sidebar για επιλογή πρωταθλήματος
st.sidebar.header("Ρυθμίσεις")
SPORT = st.sidebar.selectbox(
    "Επιλογή Πρωταθλήματος",
    [
        ("soccer_epl", "Premier League (Αγγλία)"),
        ("soccer_spain_la_liga", "La Liga (Ισπανία)"),
        ("soccer_italy_serie_a", "Serie A (Ιταλία)"),
        ("soccer_germany_bundesliga", "Bundesliga (Γερμανία)"),
        ("soccer_uefa_champs_league", "Champions League"),
    ],
    format_func=lambda x: x[1]
)[0]

def fetch_odds(api_key, sport_key):
    """Ανάκτηση αποδόσεων από το API για Ευρώπη / Κύπρο"""
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/'
    params = {
        'apiKey': api_key,
        'regions': 'eu', # Ευρωπαϊκή περιοχή (καλύπτει Κύπρο)
        'markets': 'h2h', # Αγορά 1X2
        'bookmakers': 'bet365,stoiximan_gr,stoiximan,betano_eu' # Όλα τα πιθανά κλειδιά
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Σφάλμα API ({response.status_code}): {response.text}")
        return None

if st.button("Ανανέωση Αποδόσεων 🔄"):
    if not API_KEY:
        st.warning("Παρακαλώ βεβαιωθείτε ότι έχετε ορίσει το API Key στα Secrets.")
    else:
        with st.spinner("Ανάκτηση δεδομένων..."):
            data = fetch_odds(API_KEY, SPORT)
            
            if data:
                rows = []
                for match in data:
                    home = match['home_team']
                    away = match['away_team']
                    commence_time = pd.to_datetime(match['commence_time']).strftime('%Y-%m-%d %H:%M')
                    
                    # Αρχικοποίηση τιμών
                    bet365_1, bet365_x, bet365_2 = "-", "-", "-"
                    stoiximan_1, stoiximan_x, stoiximan_2 = "-", "-", "-"
                    
                    for bookmaker in match.get('bookmakers', []):
                        bm_key = bookmaker['key']
                        for market in bookmaker.get('markets', []):
                            if market['key'] == 'h2h':
                                outcomes = {out['name']: out['price'] for out in market['outcomes']}
                                
                                h_odds = outcomes.get(home, "-")
                                a_odds = outcomes.get(away, "-")
                                d_odds = outcomes.get("Draw", "-")
                                
                                if bm_key == 'bet365':
                                    bet365_1, bet365_x, bet365_2 = h_odds, d_odds, a_odds
                                elif bm_key in ['stoiximan', 'stoiximan_gr', 'betano_eu']:
                                    stoiximan_1, stoiximan_x, stoiximan_2 = h_odds, d_odds, a_odds
                    
                    rows.append({
                        "Έναρξη": commence_time,
                        "Αγώνας": f"{home} vs {away}",
                        "Bet365 (1)": bet365_1,
                        "Stoiximan (1)": stoiximan_1,
                        "Bet365 (X)": bet365_x,
                        "Stoiximan (X)": stoiximan_x,
                        "Bet365 (2)": bet365_2,
                        "Stoiximan (2)": stoiximan_2,
                    })
                
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Δεν βρέθηκαν διαθέσιμες αποδόσεις για τις συγκεκριμένες εταιρίες.")
