import requests
from bs4 import BeautifulSoup

# Define the headers with the User-Agent string
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

url = "https://en.wikipedia.org/wiki/OpenAI"
response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.content, "html.parser")

# Save the text content to a file
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(soup.get_text())

print("Content has been saved to 'output.txt'")
