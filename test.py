import os
import psycopg2
from dotenv import load_dotenv

# Load .env file
load_dotenv()

conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()
cur.execute("SELECT * FROM users WHERE email ='22a31a4402@gmail.com'")
rows = cur.fetchall()
for row in rows:
    print(row)
cur.close()
conn.close()