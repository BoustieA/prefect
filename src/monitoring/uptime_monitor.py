#!/usr/bin/env python3
"""
Uptime Kuma Integration for IA Continu Solution
Provides uptime monitoring and health checks integration
"""

import requests
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UptimeMonitor:
    """Uptime monitoring service integration"""
    
    def __init__(self, api_url: str = "http://localhost:9000", uptime_kuma_url: str = "http://localhost:3001"):
        self.api_url = api_url
        self.uptime_kuma_url = uptime_kuma_url
        self.monitors = []
        
    def check_api_health(self) -> Dict:
        """Check API health status"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_url}/health", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "up",
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "message": "API is healthy",
                    "details": data
                }
            else:
                return {
                    "status": "down",
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "message": f"API returned status {response.status_code}"
                }
                
        except requests.RequestException as e:
            return {
                "status": "down",
                "response_time": 0,
                "status_code": 0,
                "message": f"API unreachable: {str(e)}"
            }
    
    def check_endpoint_health(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Check specific endpoint health"""
        try:
            start_time = time.time()
            url = f"{self.api_url}{endpoint}"
            
            if method.upper() == "GET":
                response = requests.get(url, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=10)
            else:
                return {"status": "error", "message": f"Unsupported method: {method}"}
            
            response_time = time.time() - start_time
            
            return {
                "status": "up" if response.status_code < 400 else "down",
                "response_time": response_time,
                "status_code": response.status_code,
                "endpoint": endpoint,
                "method": method
            }
            
        except requests.RequestException as e:
            return {
                "status": "down",
                "response_time": 0,
                "status_code": 0,
                "endpoint": endpoint,
                "message": str(e)
            }
    
    def run_comprehensive_health_check(self) -> Dict:
        """Run comprehensive health check on all endpoints"""
        logger.info("Running comprehensive health check")
        
        endpoints_to_check = [
            {"endpoint": "/health", "method": "GET"},
            {"endpoint": "/", "method": "GET"},
            {"endpoint": "/model/info", "method": "GET"},
            {"endpoint": "/datasets/list", "method": "GET"},
            {"endpoint": "/predict", "method": "POST", "data": {"features": [0.5, -0.3]}},
            {"endpoint": "/generate", "method": "POST", "data": {"samples": 100}}
        ]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "up",
            "total_endpoints": len(endpoints_to_check),
            "endpoints_up": 0,
            "endpoints_down": 0,
            "average_response_time": 0,
            "endpoint_results": []
        }
        
        total_response_time = 0
        
        for endpoint_config in endpoints_to_check:
            endpoint = endpoint_config["endpoint"]
            method = endpoint_config["method"]
            data = endpoint_config.get("data")
            
            result = self.check_endpoint_health(endpoint, method, data)
            results["endpoint_results"].append(result)
            
            if result["status"] == "up":
                results["endpoints_up"] += 1
                total_response_time += result["response_time"]
            else:
                results["endpoints_down"] += 1
                results["overall_status"] = "degraded"
        
        # Calculate average response time
        if results["endpoints_up"] > 0:
            results["average_response_time"] = total_response_time / results["endpoints_up"]
        
        # Determine overall status
        if results["endpoints_down"] == 0:
            results["overall_status"] = "up"
        elif results["endpoints_up"] == 0:
            results["overall_status"] = "down"
        else:
            results["overall_status"] = "degraded"
        
        logger.info(f"Health check completed: {results['overall_status']} - {results['endpoints_up']}/{results['total_endpoints']} endpoints up")
        
        return results
    
    def monitor_continuously(self, interval: int = 30, duration: int = None):
        """Run continuous monitoring"""
        logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        start_time = time.time()
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                logger.info(f"Monitoring cycle #{cycle_count}")
                
                # Run health check
                results = self.run_comprehensive_health_check()
                
                # Log results
                status_emoji = "✅" if results["overall_status"] == "up" else "⚠️" if results["overall_status"] == "degraded" else "❌"
                logger.info(f"{status_emoji} Cycle #{cycle_count}: {results['overall_status']} - {results['endpoints_up']}/{results['total_endpoints']} endpoints up")
                logger.info(f"   Average response time: {results['average_response_time']:.3f}s")
                
                # Check if we should stop
                if duration and (time.time() - start_time) >= duration:
                    logger.info(f"Monitoring completed after {duration} seconds")
                    break
                
                # Wait for next cycle
                logger.info(f"Waiting {interval} seconds until next cycle...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    
    def generate_uptime_report(self, cycles: int = 10) -> Dict:
        """Generate uptime report over multiple cycles"""
        logger.info(f"Generating uptime report over {cycles} cycles")
        
        report = {
            "start_time": datetime.now().isoformat(),
            "cycles": cycles,
            "results": [],
            "summary": {
                "total_checks": 0,
                "successful_checks": 0,
                "failed_checks": 0,
                "uptime_percentage": 0,
                "average_response_time": 0
            }
        }
        
        total_response_time = 0
        successful_cycles = 0
        
        for i in range(cycles):
            logger.info(f"Report cycle {i+1}/{cycles}")
            
            result = self.run_comprehensive_health_check()
            report["results"].append(result)
            
            report["summary"]["total_checks"] += result["total_endpoints"]
            report["summary"]["successful_checks"] += result["endpoints_up"]
            report["summary"]["failed_checks"] += result["endpoints_down"]
            
            if result["overall_status"] == "up":
                successful_cycles += 1
                total_response_time += result["average_response_time"]
            
            if i < cycles - 1:  # Don't sleep after last cycle
                time.sleep(5)  # Short interval for report generation
        
        # Calculate summary statistics
        if report["summary"]["total_checks"] > 0:
            report["summary"]["uptime_percentage"] = (report["summary"]["successful_checks"] / report["summary"]["total_checks"]) * 100
        
        if successful_cycles > 0:
            report["summary"]["average_response_time"] = total_response_time / successful_cycles
        
        report["end_time"] = datetime.now().isoformat()
        
        logger.info(f"Report completed: {report['summary']['uptime_percentage']:.1f}% uptime")
        
        return report
    
    def check_uptime_kuma_status(self) -> Dict:
        """Check if Uptime Kuma is running"""
        try:
            response = requests.get(self.uptime_kuma_url, timeout=5)
            return {
                "status": "running",
                "status_code": response.status_code,
                "url": self.uptime_kuma_url
            }
        except requests.RequestException as e:
            return {
                "status": "not_running",
                "error": str(e),
                "url": self.uptime_kuma_url
            }

def main():
    """Main uptime monitor execution"""
    print("📊 Uptime Monitor for IA Continu Solution")
    print("=" * 45)
    
    monitor = UptimeMonitor()
    
    print("Choose an option:")
    print("1. 🏥 Single health check")
    print("2. 📊 Comprehensive health check")
    print("3. 🔄 Continuous monitoring (30s intervals)")
    print("4. 📋 Generate uptime report (10 cycles)")
    print("5. 🔍 Check Uptime Kuma status")
    print("6. 🎯 Quick endpoint test")
    
    try:
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            print("\n🏥 Running single health check...")
            result = monitor.check_api_health()
            print(f"Status: {result['status']}")
            print(f"Response time: {result['response_time']:.3f}s")
            print(f"Message: {result['message']}")
            
        elif choice == "2":
            print("\n📊 Running comprehensive health check...")
            results = monitor.run_comprehensive_health_check()
            print(f"Overall Status: {results['overall_status']}")
            print(f"Endpoints Up: {results['endpoints_up']}/{results['total_endpoints']}")
            print(f"Average Response Time: {results['average_response_time']:.3f}s")
            
            print("\nEndpoint Details:")
            for endpoint_result in results["endpoint_results"]:
                status_emoji = "✅" if endpoint_result["status"] == "up" else "❌"
                print(f"   {status_emoji} {endpoint_result['endpoint']} ({endpoint_result['method']}) - {endpoint_result['response_time']:.3f}s")
            
        elif choice == "3":
            print("\n🔄 Starting continuous monitoring...")
            print("   Press Ctrl+C to stop")
            monitor.monitor_continuously()
            
        elif choice == "4":
            print("\n📋 Generating uptime report...")
            report = monitor.generate_uptime_report()
            print(f"Uptime: {report['summary']['uptime_percentage']:.1f}%")
            print(f"Successful checks: {report['summary']['successful_checks']}/{report['summary']['total_checks']}")
            print(f"Average response time: {report['summary']['average_response_time']:.3f}s")
            
        elif choice == "5":
            print("\n🔍 Checking Uptime Kuma status...")
            result = monitor.check_uptime_kuma_status()
            print(f"Status: {result['status']}")
            print(f"URL: {result['url']}")
            if "error" in result:
                print(f"Error: {result['error']}")
                
        elif choice == "6":
            endpoint = input("Enter endpoint to test (e.g., /health): ")
            print(f"\n🎯 Testing {endpoint}...")
            result = monitor.check_endpoint_health(endpoint)
            print(f"Status: {result['status']}")
            print(f"Response time: {result['response_time']:.3f}s")
            print(f"Status code: {result['status_code']}")
            
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Operation cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
