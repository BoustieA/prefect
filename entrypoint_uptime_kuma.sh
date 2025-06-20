#!/bin/sh

# Lancer Uptime Kuma en arrière-plan
node server/server.js &
PID=$!

# Attendre que le serveur soit prêt (exemple : loop avec curl)
echo "Waiting for Uptime Kuma to start..."
while ! curl -s http://localhost:3001/api/status > /dev/null; do
  echo "Waiting for Uptime Kuma..."
  sleep 2
done

# Créer un check via l'API REST
response=$(curl -X POST http://localhost:3001/api/checks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "FastAPI Health Check",
    "url": "http://fastapi_app:8000/health",
    "protocol": "http",
    "interval": 60,
    "active": true,
    "keyword": {
      "type": "exists",
      "value": "OK"
    }
  }')
echo "Response from Uptime Kuma API: $response"
# Garder le serveur node en avant-plan
wait $PID