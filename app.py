import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Value Betting Engine", layout="wide")

st.title("🎯 Value Betting Engine (+EV Finder)")
st.subheader("Εντοπισμός Λανθασμένων Αποδόσεων με Βάση το Fair Model της Pinnacle")

# Secrets / API Key
if "ODDS_API_KEY" in st.secrets:
    API_KEY = st.secrets["ODDS_API_KEY"]
else:
    API_KEY = "3d3e3d0ffab7cf371cb31edcad75b90a"

# Sidebar Ρυθμίσεις
st.sidebar.header("⚙️ Παράμετροι Αλγορίθμου")
TOTAL_BANKROLL = st.sidebar.number_input("Συνολικό Κεφάλαιο (€)", min_value=50, value=500, step=50)
MIN_EV = st.sidebar.slider("Ελάχιστο Απαιτούμενο Value (EV %)", min_value=1.0, max_value=15.0, value=3.0, step=0.5)
KELLY_FRACTION = st.sidebar.slider("Κλάσμα Kelly (Ρύθμιση Ρίσκου)", min_value=0.1, max_value=1.0, value=0.25, step=0.05,
                                  help="Το Fractional Kelly (π.χ. 0.25) προστατεύει το κεφάλαιο από διακυμάνσεις (variance).")

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

def remove_margin(pinnacle_odds):
    """Αφαιρεί τη γκανιότα από τις αποδόσεις της Pinnacle για να βρει τις πραγματικές πιθανότητες."""
    o1, ox, o2 = pinnacle_odds
    if o1 <= 0 or ox <= 0 or o2 <= 0:
        return None
    
    inv_sum = (1/o1) + (1/ox) + (1/o2)
    # Proportional margin removal
    true_prob_1 = (1/o1) / inv_sum
    true_prob_x = (1/ox) / inv_sum
    true_prob_2 = (1/o2) / inv_sum
    
    return true_prob_1, true_prob_x, true_prob_2

if st.button("🚀 Αναζήτηση Value Bets"):
    with st.spinner("Ανάλυση αγορών & υπολογισμός Fair Odds..."):
        data = fetch_odds(API_KEY, SPORT)
        
        if data:
            value_opportunities = []
            
            for match in data:
                home = match['home_team']
                away = match['away_team']
                commence_time = pd.to_datetime(match['commence_time']).strftime('%Y-%m-%d %H:%M')
                
                pinnacle_odds = {}
                other_bookmakers = {}
                
                # Συλλογή δεδομένων
                for bookmaker in match.get('bookmakers', []):
                    bm_key = bookmaker['key']
                    bm_title = bookmaker['title']
                    
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':
                            outcomes = {out['name']: out['price'] for out in market['outcomes']}
                            
                            if bm_key == 'pinnacle':
                                pinnacle_odds = {
                                    home: outcomes.get(home, 0),
                                    "Draw": outcomes.get("Draw", 0),
                                    away: outcomes.get(away, 0)
                                }
                            else:
                                other_bookmakers[bm_title] = {
                                    home: outcomes.get(home, 0),
                                    "Draw": outcomes.get("Draw", 0),
                                    away: outcomes.get(away, 0)
                                }
                
                # Αν υπάρχει η Pinnacle ως sharp benchmark
                if pinnacle_odds and all(pinnacle_odds.values()):
                    p_home, p_draw, p_away = remove_margin((
                        pinnacle_odds[home],
                        pinnacle_odds["Draw"],
                        pinnacle_odds[away]
                    ))
                    
                    true_probabilities = {
                        home: p_home,
                        "Draw": p_draw,
                        away: p_away
                    }
                    
                    # Έλεγχος των άλλων εταιρειών για Value
                    for bm_title, odds in other_bookmakers.items():
                        for outcome_name, offered_odd in odds.items():
                            if offered_odd > 1.0:
                                true_p = true_probabilities[outcome_name]
                                fair_odd = 1 / true_p
                                
                                # Υπολογισμός Expected Value (EV %)
                                ev_pct = ((offered_odd * true_p) - 1) * 100
                                
                                # Εάν το EV ξεπερνά το όριο που έθεσε ο χρήστης
                                if ev_pct >= MIN_EV:
                                    # Υπολογισμός Kelly Stake
                                    b = offered_odd - 1
                                    q = 1 - true_p
                                    full_kelly = (b * true_p - q) / b
                                    
                                    if full_kelly > 0:
                                        stake_pct = full_kelly * KELLY_FRACTION
                                        recommended_stake = round(TOTAL_BANKROLL * stake_pct, 2)
                                        
                                        value_opportunities.append({
                                            "Αγώνας": f"{home} vs {away}",
                                            "Έναρξη": commence_time,
                                            "Σημείο": outcome_name,
                                            "Εταιρία": bm_title,
                                            "Προσφερόμενη Απόδοση": offered_odd,
                                            "Δίκαιη Απόδοση (Fair)": round(fair_odd, 2),
                                            "Value (EV %)": f"+{ev_pct:.2f}%",
                                            "Προτεινόμενο Ποντάρισμα": f"€{recommended_stake} ({stake_pct*100:.1f}%)"
                                        })
            
            # Εμφάνιση Αποτελεσμάτων
            if value_opportunities:
                st.success(f"🎯 Βρέθηκαν {len(value_opportunities)} ευκαιρίες Value Betting!")
                df_val = pd.DataFrame(value_opportunities)
                st.dataframe(df_val, use_container_width=True)
            else:
                st.info(f"ℹ️ Δεν βρέθηκαν ευκαιρίες με EV >= +{MIN_EV}% αυτή τη στιγμή για αυτό το πρωτάθλημα.")
