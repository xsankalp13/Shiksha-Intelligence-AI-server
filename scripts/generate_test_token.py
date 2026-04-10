"""
Helper script to generate a valid test JWT for local AI service testing.
Uses the same secret as the Java backend.
Usage: python scripts/generate_test_token.py
"""
import time
from jose import jwt

# Load secret from .env manually or hardcode for dev helper
SECRET = "jfhsdafasdfasdf%af6sdf4sdaf654sd65f4sd564sd654fad5f4sa6f5sa4f65af4dsf874f65132f1sd68f5asdfbsa4n56"
ALGO = "HS256"

def generate_token(user_id=1, roles=["STUDENT"]):
    payload = {
        "sub": str(user_id),
        "userId": user_id,
        "roles": roles,
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400  # 24 hours
    }
    token = jwt.encode(payload, SECRET, algorithm=ALGO)
    return token

if __name__ == "__main__":
    t = generate_token()
    print("\n--- SHIKSHA TEST TOKEN ---")
    print(f"Bearer {t}")
    print("--------------------------\n")
    print("Copy the entire string above (including 'Bearer ') and paste it")
    print("into the 'Authorize' box in Swagger UI to test real ERP integration.\n")
