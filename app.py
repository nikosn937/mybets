import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Σύγκριση Αποδόσεων Ποδοσφαίρου", layout="wide")

st.title("⚽ Σύγκριση Αποδόσεων Ποδοσφαίρου")
st.subheader("Παρακολούθηση Αποδόσεων σε Πραγματικό Χρόνο")

# Διαβάζουμε το API Key από τα Secrets
if "ODDS_API_KEY" in st.secrets:
    API_KEY = st.secrets["ODDS_API_KEY"]
else:
    API_KEY = "3d3e3d0ffab7cf371cb31edcad75b90a" # Fallback key

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
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/'
    params = {
        'apiKey': api_key,
        'regions': 'eu,uk', # Περιλαμβάνει Bet365 και Ευρωπαϊκές εταιρίες
        'markets': 'h2h'
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Σφάλμα API ({response.status_code}): {response.text}")
        return None

if st.button("Ανανέωση Αποδόσεων 🔄"):
    with st.spinner("Ανάκτηση δεδομένων..."):
        data = fetch_odds(API_KEY, SPORT)
        
        if data:
            rows = []
            for match in data:
                home = match['home_team']
                away = match['away_team']
                commence_time = pd.to_datetime(match['commence_time']).strftime('%Y-%m-%d %H:%M')
                
                # Αρχικοποίηση για διαθέσιμες εταιρίες
                b365_1, b365_x, b365_2 = "-", "-", "-"
                pinnacle_1, pinnacle_x, pinnacle_2 = "-", "-", "-"
                unibet_1, unibet_x, unibet_2 = "-", "-", "-"
                
                for bookmaker in match.get('bookmakers', []):
                    bm_key = bookmaker['key']
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            outcomes = {out['name']: out['price'] for out in market['outcomes']}
                            
                            h = outcomes.get(home, "-")
                            a = outcomes.get(away, "-")
                            d = outcomes.get("Draw", "-")
                            
                            if bm_key == 'bet365':
                                b365_1, b365_x, b365_2 = h, d, a
                            elif bm_key == 'pinnacle':
                                pinnacle_1, pinnacle_x, pinnacle_2 = h, d, a
                            elif 'unibet' in bm_key:
                                unibet_1, unibet_x, unibet_2 = h, d, a
                
                rows.append({
                    "Έναρξη": commence_time,
                    "Αγώνας": f"{home} vs {away}",
                    "Bet365 (1)": b365_1,
                    "Bet365 (X)": b365_x,
                    "Bet365 (2)": b365_2,
                    "Pinnacle (1)": pinnacle_1,
                    "Pinnacle (X)": pinnacle_x,
                    "Pinnacle (2)": pinnacle_2,
                    "Unibet (1)": unibet_1,
                    "Unibet (X)": unibet_x,
                    "Unibet (2)": unibet_2,
                })
            
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Δεν βρέθηκαν διαθέσιμες αποδόσεις.")
