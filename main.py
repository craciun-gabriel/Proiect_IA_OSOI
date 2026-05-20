import os

def main():
    """
    Punctul de intrare principal al proiectului.
    Acest script rulează automat comanda de sistem pentru a porni 
    serverul Streamlit folosind fișierul de interfață 'app.py'.
    """
    print("Se pornește interfața grafică Streamlit...")
    os.system("streamlit run app.py")

if __name__ == "__main__":
    main()