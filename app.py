# Application Streamlit de prédiction du prix d'un trajet de taxi
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Taxi Trip Price Prediction",
    page_icon="🚕",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Thème visuel (CSS léger, uniquement ce que Streamlit ne propose pas nativement)
# Palette : fond #F6F7F9 · surface #FFFFFF · texte #1F2937 · bleu nuit #1E2A5A · indigo #3F51B5
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #F6F7F9; --surface: #FFFFFF; --text: #1F2937; --muted: #6B7280;
  --border: #E5E7EB; --primary: #1E2A5A; --accent: #3F51B5; --accent-soft: #EEF0FA;
}
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; color: var(--text); }
.stApp { background: var(--bg); }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 880px; padding-top: 2.5rem; padding-bottom: 4rem; }

/* En-tête */
.app-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.75rem; }
.app-logo { width: 52px; height: 52px; border-radius: 14px; background: var(--primary);
  display: flex; align-items: center; justify-content: center; font-size: 1.6rem; flex-shrink: 0; }
.app-title { font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; color: var(--primary); line-height: 1.2; }
.app-subtitle { margin: .2rem 0 0; color: var(--muted); font-size: .95rem; }

/* Onglets */
.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 1rem; font-weight: 500; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--primary); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent); }

/* Cartes (formulaires, conteneurs bordurés) */
[data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--surface); border: 1px solid var(--border) !important;
  border-radius: 14px; box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}
[data-testid="stForm"] { padding: 1.5rem; }

/* Titres de section dans les formulaires */
.section-title { font-size: .95rem; font-weight: 600; color: var(--primary); margin: 1.1rem 0 .15rem; }
.section-title:first-child { margin-top: 0; }
.section-hint { font-size: .85rem; color: var(--muted); margin-bottom: .6rem; }

/* Boutons */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 10px; font-weight: 600; padding: .6rem 1.4rem; transition: background .15s ease;
}
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"],
.stButton > button[kind="primary"] { background: var(--primary); border: 1px solid var(--primary); color: #fff; }
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"]:hover,
.stButton > button[kind="primary"]:hover { background: var(--accent); border-color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

/* Carte de résultat principale */
.result-card { background: var(--primary); color: #fff; border-radius: 16px; padding: 1.75rem 2rem; margin: 1.25rem 0 .75rem; }
.result-label { font-size: .9rem; color: #C7CDEB; margin: 0; }
.result-value { font-size: 2.8rem; font-weight: 700; letter-spacing: -0.02em; margin: .15rem 0 .35rem; line-height: 1.1; }
.result-note { font-size: .88rem; color: #C7CDEB; margin: 0; }

/* Puces récapitulatives */
.chips { display: flex; flex-wrap: wrap; gap: .5rem; margin: .5rem 0 1rem; }
.chip { background: var(--accent-soft); color: var(--primary); border-radius: 999px; padding: .3rem .8rem; font-size: .82rem; font-weight: 500; }

/* Indicateurs */
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .9rem 1.1rem; }
[data-testid="stMetricLabel"] { color: var(--muted); }

@media (max-width: 640px) {
  .block-container { padding-top: 1.25rem; }
  .app-title { font-size: 1.35rem; }
  .result-value { font-size: 2.2rem; }
  [data-testid="stForm"] { padding: 1rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Chargement des objets sauvegardés (une seule fois grâce au cache)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_objects():
    encoders = joblib.load("encoders.joblib")      # encodeurs
    xgb_model = joblib.load("xgb_model.joblib")    # meilleur modèle
    uniques = joblib.load("uniques.joblib")        # valeurs uniques
    scaler = joblib.load("scaler.joblib")          # normalisateur
    return encoders, xgb_model, uniques, scaler


try:
    encoders, xgb_model, uniques, scaler = load_objects()
except FileNotFoundError as e:
    st.error(
        f"Fichier introuvable : **{e.filename}**. "
        "Placez `encoders.joblib`, `xgb_model.joblib`, `uniques.joblib` et "
        "`scaler.joblib` dans le même dossier que `app.py`."
    )
    st.stop()

# Colonnes attendues, dans l'ordre attendu par le modèle
COLUMNS = [
    "Trip_Distance_km", "Time_of_Day", "Day_of_Week", "Passenger_Count",
    "Traffic_Conditions", "Weather", "Base_Fare", "Per_Km_Rate",
    "Per_Minute_Rate", "Trip_Duration_Minutes",
]


# ----------------------------------------------------------------------------
# Logique de prédiction (inchangée)
# ----------------------------------------------------------------------------
def Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
              Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes):
    # Encoder les variables catégorielles
    Time_of_Day = encoders[0].transform([Time_of_Day])[0]
    Day_of_Week = encoders[1].transform([Day_of_Week])[0]
    Traffic_Conditions = encoders[2].transform([Traffic_Conditions])[0]
    Weather = encoders[3].transform([Weather])[0]
    # Vecteur des valeurs
    x_new = np.array([Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                      Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes], dtype=float)
    x_new = x_new.reshape(1, -1)
    # Normaliser puis prédire
    x_new = scaler.transform(x_new)
    y_pred = xgb_model.predict(x_new)
    return round(float(y_pred[0]), 2)


def Pred_func_csv(df):
    """Prédit chaque ligne ; une ligne invalide reçoit NaN au lieu de bloquer tout le fichier."""
    predictions = []
    for row in df.iloc[:, :10].values:
        try:
            predictions.append(Pred_func(*row))
        except Exception:
            predictions.append(np.nan)
    df = df.copy()
    df["Trip_Price"] = predictions
    return df


# ----------------------------------------------------------------------------
# En-tête
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
      <div class="app-logo">🚕</div>
      <div>
        <h1 class="app-title">Taxi Trip Price Prediction</h1>
        <p class="app-subtitle">Estimez le prix d'un trajet à partir de la distance, des conditions
        de circulation et de la tarification.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Estimation unique", "Estimation par fichier", "À propos du modèle"])

# ----------------------------------------------------------------------------
# Onglet 1 : prédiction simple
# ----------------------------------------------------------------------------
with tab1:
    with st.form("form_simple"):
        st.markdown('<div class="section-title">Le trajet</div>'
                    '<div class="section-hint">Distance, durée et nombre de passagers.</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        Trip_Distance_km = c1.number_input("Distance (km)", min_value=0.0, value=25.0, step=0.5)
        Trip_Duration_Minutes = c2.number_input("Durée (minutes)", min_value=0.0, value=60.0, step=1.0)
        Passenger_Count = c3.number_input("Passagers", min_value=1, max_value=4, value=2, step=1)

        st.markdown('<div class="section-title">Les conditions</div>'
                    '<div class="section-hint">Moment de la journée, circulation et météo.</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        Time_of_Day = c1.selectbox("Moment de la journée", options=uniques[0])
        Day_of_Week = c2.selectbox("Jour de la semaine", options=uniques[1])
        Traffic_Conditions = c1.selectbox("Circulation", options=uniques[2])
        Weather = c2.selectbox("Météo", options=uniques[3])

        st.markdown('<div class="section-title">La tarification</div>'
                    '<div class="section-hint">Montants en $.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        Base_Fare = c1.number_input("Prise en charge ($)", min_value=0.0, value=3.5, step=0.1)
        Per_Km_Rate = c2.number_input("Tarif par km ($)", min_value=0.0, value=1.2, step=0.05)
        Per_Minute_Rate = c3.number_input("Tarif par minute ($)", min_value=0.0, value=0.3, step=0.01)

        submit = st.form_submit_button("Estimer le prix", type="primary", use_container_width=True)

    if submit:
        try:
            with st.spinner("Calcul de l'estimation…"):
                y_pred = Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count,
                                   Traffic_Conditions, Weather, Base_Fare, Per_Km_Rate,
                                   Per_Minute_Rate, Trip_Duration_Minutes)
        except Exception as e:
            st.error(f"L'estimation a échoué. Vérifiez les valeurs saisies. Détail : {e}")
        else:
            reference = Base_Fare + Trip_Distance_km * Per_Km_Rate + Trip_Duration_Minutes * Per_Minute_Rate
            st.markdown(
                f"""
                <div class="result-card" role="status">
                  <p class="result-label">Prix estimé du trajet</p>
                  <p class="result-value">{y_pred:,.2f} $</p>
                  <p class="result-note">Estimation indicative fournie par le modèle XGBoost.</p>
                </div>
                <div class="chips">
                  <span class="chip">{Trip_Distance_km:g} km</span>
                  <span class="chip">{Trip_Duration_Minutes:g} min</span>
                  <span class="chip">{Passenger_Count} passager(s)</span>
                  <span class="chip">{Time_of_Day}</span>
                  <span class="chip">{Day_of_Week}</span>
                  <span class="chip">Circulation : {Traffic_Conditions}</span>
                  <span class="chip">Météo : {Weather}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Tarif de référence", f"{reference:,.2f} $",
                      help="Prise en charge + distance × tarif/km + durée × tarif/minute.")
            m2.metric("Écart avec la prédiction", f"{y_pred - reference:+,.2f} $",
                      help="Ce que le modèle ajoute ou retire par rapport au tarif de référence "
                           "(circulation, météo, horaire…).")
    else:
        st.info("Renseignez les informations du trajet puis cliquez sur **Estimer le prix**.")

# ----------------------------------------------------------------------------
# Onglet 2 : prédiction multiple (CSV)
# ----------------------------------------------------------------------------
with tab2:
    st.markdown("Estimez plusieurs trajets d'un coup : chargez un fichier CSV, puis téléchargez les résultats.")

    with st.container(border=True):
        st.markdown('<div class="section-title">1. Préparer le fichier</div>'
                    '<div class="section-hint">Les 10 premières colonnes doivent suivre cet ordre :</div>',
                    unsafe_allow_html=True)
        st.code(", ".join(COLUMNS), language=None)

        template = pd.DataFrame([{
            "Trip_Distance_km": 25.0, "Time_of_Day": uniques[0][0], "Day_of_Week": uniques[1][0],
            "Passenger_Count": 2, "Traffic_Conditions": uniques[2][0], "Weather": uniques[3][0],
            "Base_Fare": 3.5, "Per_Km_Rate": 1.2, "Per_Minute_Rate": 0.3, "Trip_Duration_Minutes": 60.0,
        }])
        st.download_button("Télécharger un modèle de fichier", template.to_csv(index=False).encode("utf-8"),
                           file_name="modele_trajets.csv", mime="text/csv")

        st.markdown('<div class="section-title">2. Charger le fichier</div>', unsafe_allow_html=True)
        file = st.file_uploader("Fichier CSV", type=["csv"], label_visibility="collapsed")

    if file is None:
        st.session_state.pop("df_pred", None)
        st.info("Chargez un fichier CSV pour afficher un aperçu et lancer les estimations.")
    else:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Impossible de lire ce fichier CSV. Détail : {e}")
            st.stop()

        if df.shape[1] < 10:
            st.error(f"Le fichier contient {df.shape[1]} colonne(s) ; 10 sont attendues. "
                     "Téléchargez le modèle de fichier ci-dessus pour vérifier le format.")
        else:
            st.success(f"Fichier chargé : {len(df):,} trajet(s).")
            if list(df.columns[:10]) != COLUMNS:
                st.warning("Les noms de colonnes diffèrent du modèle. Les 10 premières colonnes "
                           "seront utilisées dans l'ordre indiqué ci-dessus.")
            st.markdown("**Aperçu des données**")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Lancer les estimations", type="primary"):
                with st.spinner("Calcul des estimations…"):
                    st.session_state["df_pred"] = Pred_func_csv(df)

            df_pred = st.session_state.get("df_pred")
            if df_pred is not None and len(df_pred) == len(df):
                valid = df_pred["Trip_Price"].dropna()
                n_err = int(df_pred["Trip_Price"].isna().sum())

                if n_err:
                    st.warning(f"{n_err} ligne(s) n'ont pas pu être estimées (valeur manquante, "
                               "catégorie inconnue ou nombre invalide). Elles apparaissent vides "
                               "dans la colonne Trip_Price.")
                else:
                    st.success("Toutes les estimations ont été calculées.")

                if len(valid):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Prix moyen", f"{valid.mean():,.2f} $")
                    m2.metric("Prix minimum", f"{valid.min():,.2f} $")
                    m3.metric("Prix maximum", f"{valid.max():,.2f} $")

                st.markdown("**Résultats**")
                st.dataframe(df_pred, use_container_width=True)
                st.download_button("Télécharger les résultats (CSV)",
                                   df_pred.to_csv(index=False).encode("utf-8"),
                                   file_name="predictions.csv", mime="text/csv", type="primary")

# ----------------------------------------------------------------------------
# Onglet 3 : à propos du modèle
# ----------------------------------------------------------------------------
with tab3:
    st.markdown("### Comment fonctionne l'estimation ?")
    st.write(
        "Un modèle de machine learning (XGBoost) a appris, à partir de trajets passés, comment le prix "
        "varie selon la distance, la durée, le nombre de passagers, l'horaire, la circulation, la météo "
        "et la tarification appliquée."
    )
    st.markdown("**Variables utilisées**")
    st.write(
        "Distance, durée, nombre de passagers, moment de la journée, jour de la semaine, "
        "circulation, météo, prise en charge, tarif par km et tarif par minute."
    )
    st.markdown("**Limites à connaître**")
    st.write(
        "- Le résultat est une **estimation indicative**, pas un prix garanti.\n"
        "- Les valeurs très éloignées des données d'entraînement (distances ou durées extrêmes) "
        "donnent des résultats moins fiables.\n"
        "- Le modèle ne connaît que les catégories vues à l'entraînement.\n"
        "- Les événements exceptionnels (péages, travaux, majorations ponctuelles) ne sont pas pris en compte."
    )
