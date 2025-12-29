from fastapi.testclient import TestClient

from musicrec.api.main import app


def main():
    client = TestClient(app)

    print("1) GET /feed/home")
    r1 = client.get("/feed/home", params={"country": "Brazil", "n": 5, "debug": "true"})
    print(r1.status_code, r1.json())

    print("\n2) GET /feed/genre")
    r2 = client.get("/feed/genre", params={"country": "Brazil", "genre": "Rock", "n": 5, "debug": "true"})
    print(r2.status_code, r2.json())

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()