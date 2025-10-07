# ---------- DETECTAR MODE ----------
def detect_mode(text, memory):
    t = text.lower().strip()
    # Conversa casual
    if any(x in t for x in ["hola","bon dia","bona tarda","bona nit","ei","com estàs","què et sembla"]):
        return "chat"
    # Pregunta /mes o cita directa
    elif t.startswith("/mes") or t.startswith("/") and memory.get("last_docs"):
        return "source_detail"
    # Mode resum general
    else:
        return "summary"

# ---------- FUNCIONS LLM ----------
def call_llm_chat_mode(user_query):
    prompt = [
        {"role":"system","content":"Ets una IA amable i propera del territori de la Ribera d’Ebre. Parla en català."},
        {"role":"user","content": user_query}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        temperature=0.7,
        max_tokens=200
    )
    return resp.choices[0].message.content.strip()

# ---------- HANDLE MESSAGE ----------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    m = user_memory.setdefault(chat_id, {"history": [], "last_docs": [], "last_title": None, "page": 0})

    mode = detect_mode(text, m)

    # ---------- Mode conversa casual ----------
    if mode == "chat":
        reply = call_llm_chat_mode(text)
        await update.message.reply_text(reply)
        return

    # ---------- Mode resum general ----------
    if mode == "summary":
        await update.message.reply_text("..rumiant..")
        docs_to_use = semantic_search(text, docs, embeddings, index, top_k=MAX_CONTEXT_DOCS)
        reply = call_llm_with_context(text, docs_to_use)
        # Afegir fonts utilitzades
        fonts = [f"/{i+1} {docs[i].get('title','')}" for i in docs_to_use]
        if fonts:
            reply += "\n\nFonts: " + ", ".join(fonts)
        await update.message.reply_text(reply)
        m["last_docs"] = docs_to_use
        m["last_title"] = None
        m["page"] = 0
        return

    # ---------- Mode cita directa /mes ----------
    # Si l'usuari ha escrit exactament un títol o /num
    selected_idx = None
    if text.startswith("/"):
        try:
            idx = int(text[1:]) - 1
            if 0 <= idx < len(docs):
                selected_idx = idx
        except:
            pass
    else:
        exact_idxs = find_exact_matches(text, docs)
        if exact_idxs:
            selected_idx = exact_idxs[0]

    if selected_idx is not None:
        m["last_docs"] = [selected_idx]
        m["last_title"] = docs[selected_idx].get("title")
        m["page"] = 0

        # Mostrar summary o part de summary_long amb paginació
        await send_summary_page(update, context, chat_id)
        return

# ---------- ENVIAR SUMMARY AMB PAGINACIÓ ----------
async def send_summary_page(update, context, chat_id):
    m = user_memory[chat_id]
    idx = m["last_docs"][0]
    d = docs[idx]
    long_text = d.get("summary_long", d.get("summary",""))
    max_chars = 3500
    pages = [long_text[i:i+max_chars] for i in range(0, len(long_text), max_chars)]
    page_num = m.get("page",0)
    total_pages = len(pages)
    msg = f"**{d.get('title')}**\n\n{pages[page_num]}\n\n({page_num+1}/{total_pages})"
    if page_num + 1 < total_pages:
        msg += "\n\nVols continuar? Escriu /mes"
    await update.message.reply_text(msg)
    m["page"] += 1

# ---------- HANDLER /MES ----------
async def more_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    m = user_memory.get(chat_id)
    if not m or not m.get("last_docs"):
        await update.message.reply_text("No tinc context previ. Digues-me sobre què vols informació.")
        return
    await send_summary_page(update, context, chat_id)

# ---------- HANDLER START ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola! Sóc l'assistent de la Ribera d'Ebre. "
        "Pregunta'm sobre el corpus o temes generals (riuades, guerra civil, pobles...)."
    )

# ---------- RUN BOT ----------
def run_bot():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Posa TELEGRAM_TOKEN a l'entorn")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("mes", more_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    stop_event = asyncio.Event()
    def _sigterm_handler(signum, frame):
        print("Tancant per senyal...")
        stop_event.set()
    signal.signal(signal.SIGINT, _sigterm_handler)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    print("Bot engegat. Waiting for messages...")
    app.run_polling()

if __name__ == "__main__":
    run_bot()
