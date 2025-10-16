# MisCEREbot 🤖

Bot de Telegram intel·ligent per consultes sobre la Ribera d'Ebre, desenvolupat pel Centre d'Estudis de la Ribera d'Ebre (CERE).

## ✨ Característiques

- **Cerca semàntica avançada** amb FAISS i embeddings OpenAI
- **Anàlisi intel·ligent** de consultes amb IA
- **Sistema de sessions** per mantenir context
- **Mode article** amb paginació per llegir contingut llarg
- **Filtres geogràfics** per pobles de la Ribera d'Ebre
- **Comandes avançades** (/creua, /amplia, /arxiu)
- **Síntesi automàtica** de contingut amb mode humorístic/formatiu

## 🚀 Instal·lació

### Requisits
- Python 3.8+
- Token de Telegram Bot
- Clau API d'OpenAI

### Configuració

1. **Clona el repositori:**
   ```bash
   git clone https://github.com/edx1975/CERE-RiberadEbreBOT.git
   cd CERE-RiberadEbreBOT
   ```

2. **Instal·la les dependències:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configura les variables d'entorn:**
   ```bash
   cp .env.example .env
   # Edita .env amb les teves claus API
   ```

4. **Executa el bot:**
   ```bash
   python misCEREbot.py
   ```

## 📁 Estructura del projecte

```
CERE-RiberadEbreBOT/
├── misCEREbot.py          # Bot principal
├── requirements.txt       # Dependències Python
├── Procfile              # Configuració Heroku
├── .env.example          # Exemple de variables d'entorn
├── README.md             # Documentació
└── data/                 # Dades del corpus
    ├── corpus_original.jsonl    # Corpus de documents
    ├── embeddings_G.npy         # Embeddings per cerca semàntica
    ├── faiss_index_G.index      # Índex FAISS
    ├── ajuda.txt               # Text d'ajuda
    └── expert.txt              # Text expert
```

## 🔧 Comandes del bot

### Bàsiques
- `/start` - Inicia el bot
- `/ajuda` - Mostra ajuda bàsica
- `/expert` - Mode expert avançat
- `/nou` - Reinicia la sessió

### Cerca i navegació
- `castells de la Ribera` - Cerca semàntica
- `/poble Ascó` - Filtra per poble
- `/tema arqueologia` - Cerca per tema
- `/id 123` - Obre article per ID

### Avançades
- `/arxiu` - Mostra corpus filtrat
- `/creua tema1 tema2` - Compara temes
- `/amplia` - Amplia última resposta
- `/tot` - Històric de consultes

## 🛠️ Desplegament

### Heroku
El bot està configurat per desplegar-se a Heroku amb el `Procfile` inclòs.

### Variables d'entorn necessàries
- `TELEGRAM_TOKEN` - Token del bot de Telegram
- `OPENAI_API_KEY` - Clau API d'OpenAI

## 📊 Dades

El bot utilitza un corpus de documents sobre la Ribera d'Ebre amb:
- Documents històrics i culturals
- Informació geogràfica i arqueològica
- Dades sobre pobles i monuments
- Embeddings semàntics per cerca intel·ligent

## 🤝 Contribucions

Les contribucions són benvingudes! Si vols afegir funcionalitats o millorar el bot, fes un fork i envia un pull request.

## 📄 Llicència

Aquest projecte està desenvolupat pel Centre d'Estudis de la Ribera d'Ebre (CERE).

## 📞 Contacte

Per preguntes o suport, contacta amb el CERE o obre un issue al repositori.
