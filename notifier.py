# -----------------------------------------------------
# ALERT SYSTEM: Telegram Only (Simplified)
# -----------------------------------------------------

import os
import requests

# -----------------------------------------------------
# --- TELEGRAM CONFIGURATION ---
# -----------------------------------------------------
# Replace these with your actual values
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")  # replace with your chat ID


def send_telegram_alert(msg: str):
    """Sends a message to your Telegram chat."""
    if BOT_TOKEN.startswith("YOUR_") or CHAT_ID.startswith("YOUR_"):
        print(f"--- ALERT (Not Sent) ---\n{msg}\n⚠️ Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, data=payload, timeout=5)
        response.raise_for_status()
        print("✅ Telegram alert sent successfully.")
    except requests.exceptions.Timeout:
        print("⚠️ Telegram alert ERROR: Request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Telegram alert ERROR: {e}")
    except Exception as e:
        print(f"⚠️ Unknown Telegram alert error: {e}")


# -----------------------------------------------------
# --- MAIN ALERT FUNCTION ---
# -----------------------------------------------------
def send_alert(msg: str):
    """Sends an alert via Telegram."""
    send_telegram_alert(msg)


# -----------------------------------------------------
# --- EXAMPLE TEST ---
# -----------------------------------------------------
if __name__ == "__main__":
    send_alert("🚨 Disaster Risk HIGH! Please take precautionary measures.")
