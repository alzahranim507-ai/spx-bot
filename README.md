# SPX500 Trading Bot - Hunter WICK PRO

This repository contains the cleaned Python code extracted from `spx bot full.pdf`.

## Files

- `spx_bot_full.py` - main bot code
- `requirements.txt` - Python packages needed by the bot
- `.env.example` - environment variables template

## Run

```bash
pip install -r requirements.txt
python spx_bot_full.py
```

## Required Environment Variables

Copy `.env.example` to `.env` or set these variables in your hosting platform:

```bash
TV_USERNAME=
TV_PASSWORD=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
WB_USERNAME=
WB_PASSWORD=
WB_SESSION_SECRET=change_this_to_a_long_random_secret
WB_BUDGET_MIN=50
WB_BUDGET_MAX=300
```

Do not commit real passwords, Telegram tokens, or Webull credentials to GitHub.
