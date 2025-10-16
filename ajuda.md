# MisCEREbot — Ajuda bàsica

_Disposo d’un arxiu resumit d’alguns llibres de la Miscel·lània del CERE de la Ribera d’Ebre._  
Actualment tinc documents de **Miravet, Rasquera, Tivissa, Ginestar i Benissanet**, i també de la comarca en general.

---

## Modes de cerca del BOT

- **Mode cerca general**  
  Escriu un tema o paraules clau (ex: `musulmans`, `riu Ebre`).  
  El bot et retornarà resultats rellevants.

- **Mode Llista**  
  Si dius “fes una llista de …” o “enumera …”, veuràs una llista de títols amb `/ID` per obrir-los.

- **Mode cerca article**  
  Quan activis `/n` o `/id N`, el bot fixarà un article i les consultes es faran únicament dins d’aquest text.  
  `/tot` → mostra l’article sencer, paginat  
  `/cerca` → torna al mode general

- **Mode creua** (avançat, pendent)  
  `/creua 12 23` → generarà un resum combinat dels articles 12 i 23.

---

## Mode expert / comandes avançades

| Comanda | Funció |
|--------|--------|
| `/poble NomDelPoble` | Estableix filtre pel poble. Exemple: `/poble Benissanet` |
| `/tema TextTema` | Força que la cerca se centri en aquest tema. Exemple: `/tema oficis musulmans` |
| `/n` | Entra en mode article amb l’últim article seleccionat |
| `/n23` | Obre directament l’article amb ID 23 en mode article |
| `/id 21` | Obre l’article amb ID 21 en mode article |
| `/tot` | En mode article, mostra el text complet fragmentat |
| `/mes` | En mode article, mostra la següent pàgina del text |
| `/cerca` | Tanca el mode article i torna a cercar en l’arxiu general |
| `/creua A B` | (Futur) genera una comparació / síntesi entre dos articles |

---

\*Nota: si el terminal o client Telegram suporta colors ANSI, aquests fitxers es poden adaptar perquè es vegin més vistosos. També pots mostrar aquesta ajuda amb `/ajuda` 
i `/expert` dins del bot.*

