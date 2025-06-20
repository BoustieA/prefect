import requests
from prefect import flow

API_URL = "http://api:8000/health"  # adapter au nom du service dans docker-compose

@flow
def test_api_reachable():
    response = requests.get(API_URL)
    assert response.status_code == 200
    print("API reachable. Status code:", response.status_code)

if __name__ == "__main__":
    test_api_reachable()