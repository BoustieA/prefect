import requests
import os
from pathlib import Path
from datetime import datetime, timezone
import logging
# Configure logging
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def send_discord_embed(message: str, status: str = "Succès") -> None:
    """Send a notification to Discord via Webhook when API status changes."""
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    
    if not DISCORD_WEBHOOK_URL:
        logger.warning("Discord webhook URL not configured")
        return
    
    # Color mapping for different statuses
    color_map = {
        "Succès": 5814783,  # Green
        "Erreur": 15158332,  # Red
        "Avertissement": 16776960,  # Yellow
        "Info": 3447003  # Blue
    }
    
    data = {
        "embeds": [{
            "title": "Résultats du pipeline",
            "description": message,
            "color": color_map.get(status, 3447003),
            "fields": [{
                "name": "Status",
                "value": status,
                "inline": True
            }, {
                "name": "Timestamp",
                "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "inline": True
            }],
            "footer": {
                "text": "IA Continu Solution"
            }
        }]
    }
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data, timeout=30)
        if response.status_code != 204:
            logger.error(f"Erreur lors de l'envoi de l'embed : "
                         f"{response.status_code}")
        else:
            logger.info("Embed envoyé avec succès !")
    except requests.RequestException as e:
        logger.error(f"Erreur de connexion Discord : {e}")
