import requests
import os

# --- Secrets from Render Environment Variables ---
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_alert(msg):
    """Sends a message to a Telegram chat."""
    
    # Check if secrets are missing
    if not BOT_TOKEN or not CHAT_ID:
        print(f"--- ALERT (Not Sent) ---\n{msg}\n(TELEGRAM_BOT_TOKEN or CHAT_ID not set in environment)")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown" # Optional: for *bold* or _italic_ text
    }
    
    try:
        # Add a timeout to prevent the thread from hanging
        response = requests.post(url, data=payload, timeout=5)
        response.raise_for_status() # Raise error for bad responses (4xx, 5xx)
        print(f"Telegram alert sent successfully.")
        
    except requests.exceptions.Timeout:
        print(f"Telegram alert ERROR: Request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"Telegram alert ERROR: {e}")
    except Exception as e:
        print(f"An unknown error occurred in send_alert: {e}")