import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Arbitrage / Surebet Finder", layout="wide")

st.title("⚡ Αρχείο Εντοπισμού Surebets (Arbitrage)")
st.subheader("Σύγκριση Pinnacle, Betsson & William Hill για Εγγυημένο Κέρδος")

# API Key
if "ODDS_API_KEY" in st.secrets:
    API_KEY = st.secrets["ODDS_API_KEY"]
else:
    API_KEY = "3d3e3d0ffab7cf371cb31edcad75b90a"

# Sidebar
st.sidebar.header("Ρυθμίσεις")
TOTAL_BANKROLL = st.sidebar.number_input("Συνολικό Ποσό Πονταρίσματος (€)", min_value=10, value=100, step=10)

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
        'regions': 'eu',
        'markets': 'h2h',
        'bookmakers': 'pinnacle,betsson,williamhill'
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Σφάλμα API ({response.status_code}): {response.text}")
        return None

if st.button("Αναζήτηση Ευκαιριών Surebet 🔍"):
    with st.spinner("Υπολογισμός αποδόσεων & έλεγχος για Arbitrage..."):
        data = fetch_odds(API_KEY, SPORT)
        
        if data:
            surebets_found = []
            all_matches = []
            
            for match in data:
                home = match['home_team']
                away = match['away_team']
                commence_time = pd.to_datetime(match['commence_time']).strftime('%Y-%m-%d %H:%M')
                
                # Αποθήκευση όλων των τιμών ανά bookmaker
                # Structure: { '1': [(price, bookmaker)], 'X': [...], '2': [...] }
                best_1 = (0, "-")
                best_x = (0, "-")
                best_2 = (0, "-")
                
                for bookmaker in match.get('bookmakers', []):
                    bm_name = bookmaker['title']
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            for outcome in market['outcomes']:
                                name = outcome['name']
                                price = outcome['price']
                                
                                if name == home and price > best_1[0]:
                                    best_1 = (price, bm_name)
                                elif name == "Draw" and price > best_x[0]:
                                    best_x = (price, bm_name)
                                elif name == away and price > best_2[0]:
                                    best_2 = (price, bm_name)
                
                # Αν βρέθηκαν τιμές και για τα 3 σημεία
                if best_1[0] > 0 and best_x[0] > 0 and best_2[0] > 0:
                    implied_prob = (1 / best_1[0]) + (1 / best_x[0]) + (1 / best_2[0])
                    profit_pct = (1 / implied_prob - 1) * 100
                    
                    match_info = {
                        "Αγώνας": f"{home} vs {away}",
                        "Έναρξη": commence_time,
                        "Καλύτερος 1": f"{best_1[0]} ({best_1[1]})",
                        "Καλύτερο X": f"{best_x[0]} ({best_x[1]})",
                        "Καλύτερο 2": f"{best_2[0]} ({best_2[1]})",
                        "Γκανιότα / Prob": f"{implied_prob*100:.2f}%",
                        "Περιθώριο / Profit": round(profit_pct, 2)
                    }
                    
                    all_matches.append(match_info)
                    
                    # Αν η συνολική πιθανότητα είναι < 1.0 (δηλαδή profit > 0), έχουμε Surebet!
                    if implied_prob < 1.0:
                        # Υπολογισμός πονταρισμάτων
                        stake_1 = round((TOTAL_BANKROLL / best_1[0]) / implied_prob, 2)
                        stake_x = round((TOTAL_BANKROLL / best_x[0]) / implied_prob, 2)
                        stake_2 = round((TOTAL_BANKROLL / best_2[0]) / implied_prob, 2)
                        guaranteed_payout = round(stake_1 * best_1[0], 2)
                        guaranteed_profit = round(guaranteed_payout - TOTAL_BANKROLL, 2)
                        
                        surebets_found.append({
                            "Αγώνας": f"{home} vs {away}",
                            "Κέρδος %": f"+{profit_pct:.2f}%",
                            "Σίγουρο Κέρδος (€)": f"€{guaranteed_profit}",
                            "Ποντάρισμα (1)": f"€{stake_1} στο {best_1[0]} ({best_1[1]})",
                            "Ποντάρισμα (X)": f"€{stake_x} στο {best_x[0]} ({best_x[1]})",
                            "Ποντάρισμα (2)": f"€{stake_2} στο {best_2[0]} ({best_2[1]})",
                        })
            
            # Εμφάνιση Αποτελεσμάτων
            if surebets_found:
                st.success(f"🎉 Βρέθηκαν {len(surebets_found)} ευκαιρίες Surebet!")
                st.dataframe(pd.DataFrame(surebets_found), use_container_width=True)
            else:
                st.warning("⚠️ Δεν βρέθηκε κανένα Surebet με θετικό κέρδος αυτή τη στιγμή ανάμεσα σε αυτές τις 3 εταιρίες.")
            
            st.divider()
            st.subheader("📊 Όλοι οι Αγώνες & Καλύτερες Συνδυαστικές Αποδόσεις")
            if all_matches:
                df_all = pd.DataFrame(all_matches)
                st.dataframe(df_all, use_container_width=True)
