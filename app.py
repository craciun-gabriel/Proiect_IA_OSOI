"""
Aplicație IA pe Echipă - Platformă de Optimizare și Procesare a Limbajului Natural (NLP)
Interfață grafică completă realizată în Streamlit.

Acest fișier centralizează și parametrizează complet:
1. Modulul TSP: Rulare individuală și benchmark comparativ global pentru BKT, NN, HC, SA, GA.
2. Modulul NLP: Antrenare și evaluare pe dataset-uri EXTINSE REALE în engleză (IMDB și SMS Spam).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from src.utils.io_utils import genereaza_matrice_aleatorie
from src.backtracking.backtracking import rezolva_tsp_backtracking
from src.nearest_neighbor.nearest_neighbor import rezolva_tsp_nn_multistart
from src.hill_climbing.hill_climbing_tsp import rezolva_tsp_hc
from src.simulated_annealing.annealer import SimulatedAnnealingTSP
from src.genetic_algorithm.tsp_genetic import GATSP
from src.nlp_classification.nlp_classification import evalueaza_nlp

st.set_page_config(page_title="Proiect IA", layout="wide")

modul_principal = st.sidebar.radio("Selectați Modulul Aplicației:", ["🗺️ Optimizare Probleme Combinatorice (TSP)", "🔤 Clasificare Limbaj Natural (NLP)"])

# ==========================================
# MODULUL 1: PROBLEMA COMIS-VOIAJORULUI (TSP)
# ==========================================
if modul_principal == "🗺️ Optimizare Probleme Combinatorice (TSP)":
    st.title("🗺️ Optimizare TSP")
    st.markdown("Implementarea, parametrizarea și compararea vizuală a algoritmilor BKT, NN, HC, SA și GA.")
    
    tab1, tab2 = st.tabs(["🎛️ Rulare și Parametrizare Individuală", "📊 Dashboard Comparativ Global"])
    
    if 'n_orase_curent' not in st.session_state:
        st.session_state.n_orase_curent = 12
        st.session_state.matrice_curenta = genereaza_matrice_aleatorie(12)
        
    with tab1:
        st.subheader("Configurare Instanță și Algoritm")
        col_st, col_dr = st.columns([1, 2])
        
        with col_st:
            n_orase = st.number_input("Număr de orașe (N):", min_value=4, max_value=150, value=st.session_state.n_orase_curent)
            if n_orase != st.session_state.n_orase_curent:
                st.session_state.n_orase_curent = n_orase
                st.session_state.matrice_curenta = genereaza_matrice_aleatorie(n_orase)
                
            algoritm = st.selectbox("Alegeți Algoritmul de Optimizare:", [
                "Backtracking (BKT)", 
                "Nearest Neighbor (NN-Multistart)", 
                "Hill Climbing (HC)", 
                "Simulated Annealing (SA)", 
                "Algoritm Genetic (GA)"
            ])
            
            st.markdown("---")
            st.markdown("#### Parametri Specifici Algoritmului")
            
            params = {}
            if algoritm == "Backtracking (BKT)":
                params['mod'] = st.selectbox("Mod Oprire:", ["toate", "prima", "timp", "y_solutii"])
                params['timp_max'] = st.slider("Limită timp (secunde):", 1, 60, 10)
                params['y_max'] = st.number_input("Număr maxim soluții (Y):", min_value=1, max_value=1000, value=10)
                st.info("⚠️ BKT este exhaustiv. Pentru N > 12 modul 'toate' poate bloca aplicația.")
                
            elif algoritm == "Nearest Neighbor (NN-Multistart)":
                st.info("Algoritm constructive Greedy. Rulează automat din fiecare oraș ca punct de pornire pentru a extrage optimul.")
                
            elif algoritm == "Hill Climbing (HC)":
                params['restarts'] = st.slider("Număr de Restarturi Aleatoare:", 1, 50, 15)
                
            elif algoritm == "Simulated Annealing (SA)":
                params['t_max'] = st.number_input("Temperatura Inițială (T_max):", min_value=100, max_value=100000, value=10000)
                params['t_min'] = st.number_input("Temperatura Minimă (T_min):", min_value=1, max_value=100, value=1)
                params['alpha'] = st.slider("Rata de Răcire Geometrică (Alpha):", 0.80, 0.99, 0.95, step=0.01)
                params['iters'] = st.slider("Iterații per Temperatură:", 10, 500, 100)
                
            elif algoritm == "Algoritm Genetic (GA)":
                params['pop_size'] = st.slider("Dimensiunea Populației:", 20, 500, 100)
                params['generations'] = st.slider("Număr de Generații:", 10, 1000, 200)
                params['mutation_rate'] = st.slider("Rata de Mutație (%):", 1, 100, 40)
                
            st.markdown("---")
            buton_ruleaza = st.button("🚀 Execută Algoritmul")
            
        with col_dr:
            if buton_ruleaza:
                matrice = st.session_state.matrice_curenta
                traseu, cost, timp, istoric = [], 0, 0.0, []
                
                with st.spinner(f"Se execută {algoritm}..."):
                    if algoritm == "Backtracking (BKT)":
                        traseu, cost, nr_sol, timp = rezolva_tsp_backtracking(n_orase, matrice, mod=params['mod'], timp_max=params['timp_max'], y_max=params['y_max'])
                    elif algoritm == "Nearest Neighbor (NN-Multistart)":
                        traseu, cost, timp = rezolva_tsp_nn_multistart(n_orase, matrice)
                    elif algoritm == "Hill Climbing (HC)":
                        traseu, cost, timp = rezolva_tsp_hc(n_orase, matrice, restarts=params['restarts'])
                    elif algoritm == "Simulated Annealing (SA)":
                        sa = SimulatedAnnealingTSP(matrice, t_max=params['t_max'], t_min=params['t_min'], alpha=params['alpha'], iters_per_temp=params['iters'])
                        traseu, cost, istoric, timp = sa.solve()
                    elif algoritm == "Algoritm Genetic (GA)":
                        ga = GATSP(matrice, pop_size=params['pop_size'], generations=params['generations'], mutation_rate=params['mutation_rate'])
                        traseu, cost, istoric, timp = ga.solve()
                        
                st.markdown("### 📈 Rezultatele Rulării")
                c1, c2, c3 = st.columns(3)
                c1.metric("Cel mai bun Cost (Distanță)", f"{cost} unități")
                c2.metric("Timp de Execuție", f"{timp:.4f} sec")
                c3.metric("Dimensiune Problemă (N)", f"{n_orase} orașe")
                
                st.code(f"Traseu Optim Găsit: {traseu}", language="python")
                
                if len(istoric) > 0:
                    st.markdown("#### Curba de Convergență (Evoluția Costului)")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(istoric, color='#1f77b4', linewidth=2, label="Cost minim curent")
                    ax.set_xlabel("Iterații / Generații")
                    ax.set_ylabel("Cost Traseu")
                    ax.grid(True, linestyle="--", alpha=0.6)
                    ax.legend()
                    st.pyplot(fig)
            else:
                st.info("Alegeți parametrii din partea stângă și apăsați butonul 'Execută Algoritmul' pentru a vizualiza performanța.")
                
            st.markdown("#### Structura Instanței Curente (Heatmap-ul Matricii de Distanțe)")
            fig_map, ax_map = plt.subplots(figsize=(10, 4))
            sns.heatmap(st.session_state.matrice_curenta, cmap="YlOrRd", ax=ax_map, annot=n_orase <= 15)
            st.pyplot(fig_map)

    with tab2:
        st.subheader("Benchmark Comparativ Simultane (Toți Algoritmii)")
        st.markdown("Rulează toți algoritmii simultan pe exact aceeași matrice de distanțe pentru a genera grafice de comparație directă a performanței (Timp vs Calitate Soluție).")
        
        n_bench = st.slider("Alegeți N pentru Benchmark:", 5, 12, 10, help="N este limitat la maxim 12 pentru a permite rularea Backtracking-ului fără blocaje.")
        
        if st.button("📊 Pornește Comparația Globală"):
            matrice_bench = genereaza_matready = genereaza_matrice_aleatorie(n_bench)
            rezultate_globale = {}
            
            with st.spinner("Se rulează benchmark-ul pentru toți algoritmii..."):
                t_b, c_b, _, timp_b = rezolva_tsp_backtracking(n_bench, matrice_bench, mod='toate')
                rezultate_globale["Backtracking"] = {"Cost": c_b, "Timp": timp_b}
                
                _, c_nn, timp_nn = rezolva_tsp_nn_multistart(n_bench, matrice_bench)
                rezultate_globale["Nearest Neighbor"] = {"Cost": c_nn, "Timp": timp_nn}
                
                _, c_hc, timp_hc = rezolva_tsp_hc(n_bench, matrice_bench, restarts=10)
                rezultate_globale["Hill Climbing"] = {"Cost": c_hc, "Timp": timp_hc}
                
                sa_b = SimulatedAnnealingTSP(matrice_bench, t_max=5000, alpha=0.90, iters_per_temp=50)
                _, c_sa, _, timp_sa = sa_b.solve()
                rezultate_globale["Simulated Annealing"] = {"Cost": c_sa, "Timp": timp_sa}
                
                ga_b = GATSP(matrice_bench, pop_size=50, generations=100, mutation_rate=30)
                _, c_ga, _, timp_ga = ga_b.solve()
                rezultate_globale["Genetic Algorithm"] = {"Cost": c_ga, "Timp": timp_ga}
                
            df_res = pd.DataFrame(rezultate_globale).T
            st.dataframe(df_res)
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown("#### Comparație Cost Traseu (Mai mic = Mai bun)")
                fig_c, ax_c = plt.subplots()
                sns.barplot(x=df_res.index, y=df_res["Cost"], palette="viridis", ax=ax_c)
                plt.xticks(rotation=45)
                st.pyplot(fig_c)
                
            with col_g2:
                st.markdown("#### Comparație Timp de Execuție (Log scale pentru claritate)")
                fig_t, ax_t = plt.subplots()
                sns.barplot(x=df_res.index, y=df_res["Timp"], palette="magma", ax=ax_t)
                ax_t.set_yscale('log')
                plt.xticks(rotation=45)
                st.pyplot(fig_t)

# ==========================================
# MODULUL 2: PROCESAREA LIMBAJULUI NATURAL (NLP)
# ==========================================
elif modul_principal == "🔤 Clasificare Limbaj Natural (NLP)":
    st.title("🔤 Clasificator NLP Parametrizabil pe Date Reale")
    st.markdown("Explorarea algoritmilor NLP pe dataset-uri extinse, complexe, exclusiv în limba engleză.")

    @st.cache_data
    def incarca_dataset_real(tip_dataset):
        try:
            if tip_dataset == "IMDB Movie Reviews (Sentiment Analysis)":
                cale = "data/IMDB Dataset.csv"
                if not os.path.exists(cale):
                    return None
                df = pd.read_csv(cale)
                df = df.rename(columns={'review': 'text', 'sentiment': 'label'})
                return df
                
            elif tip_dataset == "SMS Spam Collection (Spam/Ham)":
                cale = "data/SMSSpamCollection.csv"
                if not os.path.exists(cale):
                    return None
                
                import csv
                df = pd.read_csv(cale, sep='\t', names=['label', 'text'], header=None, quoting=csv.QUOTE_NONE)
                
                if df['text'].isnull().all():
                    df = pd.read_csv(cale, header=None, names=['raw'], quoting=csv.QUOTE_NONE)
                    df[['label', 'text']] = df['raw'].str.split('\t', n=1, expand=True)
                    df = df.drop(columns=['raw'])
                    
                df = df[['text', 'label']]
                return df
                
        except Exception as e:
            st.error(f"Eroare tehnică la citirea fișierului: {e}")
            return None
        return None

    col_nlp_st, col_nlp_dr = st.columns([1, 2])
    
    with col_nlp_st:
        st.subheader("Configurare Model și Pipeline")
        dataset_ales = st.selectbox("Selectați Dataset-ul Extins (Engleză):", [
            "IMDB Movie Reviews (Sentiment Analysis)",
            "SMS Spam Collection (Spam/Ham)"
        ])
        
        df_complet = incarca_dataset_real(dataset_ales)
        
        if df_complet is not None:
            total_randuri_disponibile = len(df_complet)
            st.success(f"Fișier detectat cu succes! Conține {total_randuri_disponibile} înregistrări.")
            
            nr_randuri = st.slider("Dimensiune subset pentru antrenare:", 
                                   min_value=500, 
                                   max_value=min(total_randuri_disponibile, 20000), 
                                   value=4000, 
                                   step=500,
                                   help="Se recomandă între 3000 și 5000 de rânduri pentru o execuție rapidă și stabilă în direct.")
            
            # Extragerea eșantionului stabil din setul de date
            df_nlp = df_complet.sample(n=nr_randuri, random_state=42).reset_index(drop=True)

            df_nlp = df_nlp.dropna(subset=['text', 'label'])
            df_nlp['text'] = df_nlp['text'].astype(str)
        else:
            st.error(f"❌ Nu s-a găsit fișierul corect în directorul 'data/'. Asigurați-vă că fișierele se numesc exact 'IMDB Dataset.csv' sau 'SMSSpamCollection.csv'.")
            st.stop()
            
        model_ales = st.selectbox("Selectați Clasificatorul ML:", ["LinearSVC", "Naive Bayes", "Random Forest"])
        
        st.markdown("#### Parametrizare TF-IDF Vectorizer")
        ngram_range_idx = st.selectbox("N-gram Range:", ["Unigrams (1, 1)", "Bigrams (1, 2)", "Trigrams (1, 3)"])
        ngram_map = {"Unigrams (1, 1)": (1, 1), "Bigrams (1, 2)": (1, 2), "Trigrams (1, 3)": (1, 3)}
        ngram_range = ngram_map[ngram_range_idx]
        
        max_features = st.slider("Max Features (Vocabular limitat):", min_value=100, max_value=5000, value=2000, step=100)
        
        st.markdown("---")
        buton_nlp = st.button("🧠 Antrenează și Evaluează Modelul")
        
    with col_nlp_dr:
        st.subheader(f"Avanpremieră Date Reale: {dataset_ales}")
        st.dataframe(df_nlp.head(5), use_container_width=True)
        
        if buton_nlp:
            # Împărțirea datelor în Train (70%) și Test (30%)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(df_nlp['text'], df_nlp['label'], test_size=0.3, random_state=42)
            
            with st.spinner("Se construiește vectorizarea TF-IDF și se rulează procesul de antrenare..."):
                pipeline, acc, cm, report, timp_nlp = evalueaza_nlp(X_train, y_train, X_test, y_test, model_ales, ngram_range)
                
            st.session_state.pipeline_antrenat = pipeline
            st.session_state.clase_nlp = list(pipeline.classes_)
            
            st.success(f"🚀 Model antrenat pe bune în {timp_nlp:.4f} secunde!")
            st.metric("Acuratețea Generală a Modelului (Accuracy)", f"{acc*100:.2f}%")
            
            # Afișare raport detaliat de metrici sub formă de tabel 
            df_report = pd.DataFrame(report).transpose().iloc[:-3, :3]
            st.markdown("#### Raport de Clasificare Detaliat (Precizie, Recall, F1)")
            st.dataframe(df_report.style.format("{:.4f}"))
            
            # Generare matrice de confuzie grafică prin Seaborn
            st.markdown("#### Matricea de Confuzie")
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=pipeline.classes_, yticklabels=pipeline.classes_, ax=ax_cm)
            ax_cm.set_xlabel("Etichetă Prezistă")
            ax_cm.set_ylabel("Etichetă Reală")
            st.pyplot(fig_cm)
        else:
            st.info("Apăsați pe 'Antrenează și Evaluează Modelul' pentru a procesa seturile masive de date.")

    st.markdown("---")
    st.subheader("🔮 Modul de Testare în Timp Real (Inference Live)")
    st.markdown("Introduceți un text propriu (în limba engleză) pentru a vedea cum îl clasifică modelul antrenat anterior.")
    
    text_utilizator = st.text_area("Introduceți textul în engleză aici:", placeholder="Type a custom movie review or a random message to test the classifier...")
    
    if st.button("🔮 Prezice Categoria"):
        if 'pipeline_antrenat' in st.session_state:
            if text_utilizator.strip() == "":
                st.warning("Vă rugăm să introduceți un text valid.")
            else:
                predictie = st.session_state.pipeline_antrenat.predict([text_utilizator])[0]
                st.success(f"Rezultat Predicție: **{predictie}**")
        else:
            st.error("❌ Trebuie să antrenați modelul mai întâi (folosind butonul de mai sus) înainte de a face predicții live.")