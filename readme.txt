=============================
GINESTARBOT - GUIA D'USUARI
=============================

1️⃣ Processar PDFs a JSONL (pdf2ai.py)

Aquest script recull els PDFs, extreu metadades i fragments de text,
i guarda tot en un fitxer JSONL que després és utilitzat pel bot o
per generar embeddings.

Funcionalitats:
- Processa tots els PDFs d’una carpeta.
- Permet processar només els nous PDFs o tornar a processar tots.
- Extreu metadades:
  - Títol
  - Subtítol
  - Autors
  - Any de publicació
  - Poble
  - Idioma
  - Paraules clau / tags
- Divideix el text en fragments (chunks).

Com usar:
1. Activar l’entorn Python:
   conda activate ginestar

2. Executar l’script:
   python pdf2ai.py

3. Opcions al principi:
   - Processar tots els PDFs de nou
   - Processar només els nous PDFs

Resultat:
- Fitxer JSONL: data/corpus_complet.jsonl
- Cada línia conté un fragment i la seva metadata

------------------------------------------------------------

2️⃣ Regenerar embeddings i índex FAISS (regenera.py)

Aquest script crea embeddings i índex FAISS a partir del JSONL,
perquè el bot amb IA pugui recuperar fragments rellevants.

Funcionalitats:
- Carrega el JSONL de fragments
- Genera embeddings amb SentenceTransformer
- Guarda embeddings a data/embeddings.npy
- Crea índex FAISS i el guarda a data/faiss.index
- Revisa nombre de fragments i embeddings

Com usar:
1. Activar l’entorn:
   conda activate ginestar

2. Executar:
   python regenera.py

Resultat:
- Embeddings: data/embeddings.npy
- Índex FAISS: data/faiss.index

------------------------------------------------------------

3️⃣ Bot de Telegram (GinestarBOT.py)

El bot interactua amb l’usuari, amb o sense IA.

Funcionalitats:
1. Comandes:
   - /start → benvinguda i instruccions
   - /list → llista d’articles amb títol, poble i any

2. Preguntes directes:
   - Qualsevol text que no sigui comanda es considera una pregunta
   - Recupera fragments més rellevants amb FAISS + embeddings
   - Retorna resum breu (~400 caràcters)
   - Botons inline:
     - Ampliar informació (~1000 caràcters)
     - Fer nova pregunta
     - Veure referències (títols i anys)

3. Mode IA:
   - Mock / simulació (USE_MOCK=True) → prova sense tokens
   - OpenAI real (USE_MOCK=False) → GPT-3.5 o GPT-4

4. Historial per usuari:
   - Manté preguntes i respostes
   - Permet ampliar informació i veure referències

5. Filtratge opcional:
   - Per metadades: any, poble, títol, etc.

Preparació:
1. Activar entorn:
   conda activate ginestar

2. Assegurar que tens:
   - data/corpus_complet.jsonl
   - data/embeddings.npy
   - data/faiss.index
   - .env amb:
     TELEGRAM_TOKEN=XXX
     OPENAI_API_KEY=XXX  # si USE_MOCK=False

3. Executar:
   python GinestarBOT.py

4. Configuració dins del codi:
   - USE_MOCK=True → prova local sense crèdit
   - USE_MOCK=False → OpenAI amb saldo
   - IA_MODEL → 3.5 o 4 segons necessitat

------------------------------------------------------------

4️⃣ Flux recomanat per producció

1. Afegir nous PDFs a la carpeta pdfs/
2. Executar pdf2ai.py:
   - Només nous PDFs per evitar duplicació
3. Executar regenera.py:
   - Actualitzar embeddings i índex FAISS
4. Llançar GinestarBOT.py:
   - Mode simulació per proves (USE_MOCK=True)
   - Mode real per producció (USE_MOCK=False)

------------------------------------------------------------

5️⃣ Consells i bones pràctiques

- Fer backup de JSONL i embeddings abans de regenerar
- Comprovar logs per errors d’IA o fragments mal processats
- Rate limits:
  - GPT-4 → revisar límits de tokens/minut
  - GPT-3.5 → més segur per molts fragments
- Historial i respostes:
  - Reiniciar el bot es perd historial → considerar persistència
- Simulació (USE_MOCK=True) → molt útil per test abans de gastar tokens
- Logging i control de errors de rate-limit: útil per producció

------------------------------------------------------------

Opcional:
- Es pot generar un diagrama visual del flux:
  PDFs → JSONL → Embeddings → FAISS → Bot → Respostes IA

