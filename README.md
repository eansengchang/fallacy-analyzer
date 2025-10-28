# 🧠 Discord Gemini AI Bot

A Discord bot powered by **Google Gemini 2.5 Flash**, built using `discord.py` and `aiohttp`.  
It provides advanced text analysis tools — including logical fallacy detection, grammar correction, summarization, and neutral discussion solutions — directly within your Discord server.

---

## 🚀 Features

- **🧩 Logical Fallacy Detection** — Detects reasoning errors in text (`e analyse` or `e analyze`).
- **✍️ Grammar Checker** — Identifies grammatical mistakes and suggests corrections (`e grammar`).
- **📝 TL;DR Summarizer** — Summarizes a conversation from a replied message (`e tldr`).
- **🤝 Neutral Solution Generator** — Suggests unbiased resolutions for discussions (`e solution`).
- **💬 Snipe / EditSnipe** — Retrieves deleted (`e snipe`) or edited (`e editsnipe`) messages.

---

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/discord-gemini-bot.git
cd discord-gemini-bot
```

### 2. Install Dependencies (via Poetry)
```bash
poetry install
```

### 3. Environment Variables
Create a `.env` file in the root directory with:
```
DISCORD_TOKEN=your_discord_bot_token
GEMINI_KEY=your_google_gemini_api_key
```

### 4. Run the Bot
```bash
poetry run python main.py
```

---

## 🧩 Commands Overview

| Command | Alias | Description |
|----------|--------|-------------|
| `e analyse [text]` | `e analyze` | Analyses text for logical fallacies. |
| `e grammar [text]` | — | Detects grammar/spelling mistakes. |
| `e tldr` | `e summarise`, `e summarize` | Summarizes a conversation thread. |
| `e solution` | — | Suggests a neutral, actionable solution. |
| `e snipe [n]` | — | Displays deleted messages (up to last 10). |
| `e editsnipe [n]` | — | Displays edited messages (up to last 10). |

---

## 🧰 Tech Stack
- **Language:** Python 3.10+
- **Package Manager:** Poetry
- **Libraries:** `discord.py`, `aiohttp`, `python-dotenv`
- **AI Model:** Google **Gemini 2.5 Flash**
- **Architecture:** Asynchronous / Modular Cog System

---

## 📜 License
MIT License — free to use and modify with attribution.
