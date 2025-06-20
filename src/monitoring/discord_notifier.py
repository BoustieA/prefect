#!/usr/bin/env python3
"""
Discord Notification Service for IA Continu Solution
Handles Discord webhook notifications with proper formatting
"""

import os
import requests
import logging
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiscordNotifier:
    """Discord webhook notification service"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        
    def send_notification(self, message: str, status: str = "Succès") -> bool:
        """Send notification to Discord webhook with Day 1 format"""
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False
        
        # Color mapping
        color_map = {
            "Succès": 5814783,    # Green
            "Échec": 15158332,    # Red
            "Avertissement": 16776960,  # Yellow
            "Info": 3447003       # Blue
        }
        
        color = color_map.get(status, 3447003)
        
        data = {
            "embeds": [{
                "title": "Résultats du pipeline",
                "description": message,
                "color": color,
                "fields": [{
                    "name": "Status",
                    "value": status,
                    "inline": True
                }, {
                    "name": "Timestamp",
                    "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "inline": True
                }],
                "footer": {
                    "text": "IA Continu Solution - Day 2"
                }
            }]
        }
        
        try:
            response = requests.post(self.webhook_url, json=data, timeout=10)
            if response.status_code == 204:
                logger.info(f"✅ Discord notification sent: {message}")
                return True
            else:
                logger.warning(f"❌ Discord notification failed: {response.status_code}")
                logger.warning(f"Response: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Discord notification error: {e}")
            return False
    
    def test_webhook(self) -> bool:
        """Test Discord webhook functionality"""
        if not self.webhook_url:
            logger.error("Discord webhook URL not configured")
            return False
        
        test_message = "🧪 Test notification from IA Continu Solution"
        logger.info("Testing Discord webhook...")
        
        success = self.send_notification(test_message, "Info")
        
        if success:
            logger.info("✅ Discord webhook test successful")
        else:
            logger.error("❌ Discord webhook test failed")
        
        return success
    
    def send_pipeline_success(self, details: str) -> bool:
        """Send pipeline success notification"""
        message = f"✅ Pipeline executed successfully\n{details}"
        return self.send_notification(message, "Succès")
    
    def send_pipeline_failure(self, error: str) -> bool:
        """Send pipeline failure notification"""
        message = f"🔴 Pipeline execution failed\nError: {error}"
        return self.send_notification(message, "Échec")
    
    def send_model_retrain_success(self, model_version: str, accuracy: float, samples: int) -> bool:
        """Send model retraining success notification"""
        message = f"✅ Model Retraining Successful\nVersion: {model_version}\nAccuracy: {accuracy:.3f}\nTraining Samples: {samples}"
        return self.send_notification(message, "Succès")
    
    def send_model_retrain_failure(self, error: str) -> bool:
        """Send model retraining failure notification"""
        message = f"🔴 Model Retraining Failed\nError: {error}"
        return self.send_notification(message, "Échec")
    
    def send_monitoring_alert(self, alert_type: str, details: str) -> bool:
        """Send monitoring alert notification"""
        if alert_type.lower() in ["error", "critical", "failure"]:
            status = "Échec"
            emoji = "🚨"
        elif alert_type.lower() in ["warning", "caution"]:
            status = "Avertissement"
            emoji = "⚠️"
        else:
            status = "Info"
            emoji = "ℹ️"
        
        message = f"{emoji} {alert_type.title()} Alert\n{details}"
        return self.send_notification(message, status)

def test_discord_webhook():
    """Test Discord webhook functionality"""
    print("🧪 Testing Discord Webhook")
    print("=" * 30)
    
    # Check if webhook URL is configured
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("❌ DISCORD_WEBHOOK_URL environment variable not set")
        print("Please set it with your Discord webhook URL:")
        print("export DISCORD_WEBHOOK_URL='your_webhook_url'")
        return False
    
    print(f"📡 Webhook URL configured: {webhook_url[:50]}...")
    
    # Initialize notifier
    notifier = DiscordNotifier(webhook_url)
    
    # Test basic notification
    print("\n1. Testing basic notification...")
    success1 = notifier.test_webhook()
    
    # Test success notification
    print("\n2. Testing success notification...")
    success2 = notifier.send_pipeline_success("Test pipeline execution completed")
    
    # Test failure notification
    print("\n3. Testing failure notification...")
    success3 = notifier.send_pipeline_failure("Test error message")
    
    # Test model retraining notification
    print("\n4. Testing model retraining notification...")
    success4 = notifier.send_model_retrain_success("v20250617_test", 0.95, 1000)
    
    # Test monitoring alert
    print("\n5. Testing monitoring alert...")
    success5 = notifier.send_monitoring_alert("Warning", "Test monitoring alert")
    
    # Summary
    total_tests = 5
    passed_tests = sum([success1, success2, success3, success4, success5])
    
    print(f"\n📊 Test Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("✅ All Discord webhook tests passed!")
        return True
    else:
        print("❌ Some Discord webhook tests failed")
        return False

def main():
    """Main Discord notifier execution"""
    print("📱 Discord Notification Service")
    print("=" * 35)
    
    # Check configuration
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if webhook_url:
        print("✅ Discord webhook URL configured")
    else:
        print("❌ Discord webhook URL not configured")
        print("Set DISCORD_WEBHOOK_URL environment variable")
        return
    
    notifier = DiscordNotifier()
    
    print("\nChoose an option:")
    print("1. Test webhook")
    print("2. Send test notification")
    print("3. Send success notification")
    print("4. Send failure notification")
    print("5. Run comprehensive test")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            print("\n🧪 Testing webhook...")
            notifier.test_webhook()
            
        elif choice == "2":
            message = input("Enter test message: ")
            notifier.send_notification(message, "Info")
            
        elif choice == "3":
            details = input("Enter success details: ")
            notifier.send_pipeline_success(details)
            
        elif choice == "4":
            error = input("Enter error message: ")
            notifier.send_pipeline_failure(error)
            
        elif choice == "5":
            print("\n🔄 Running comprehensive test...")
            test_discord_webhook()
            
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Operation cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
