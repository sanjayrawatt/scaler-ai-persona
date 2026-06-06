import urllib.request
import json

url = "https://api.github.com/users/sanjayrawatt/repos?per_page=100&type=public"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        repos = json.loads(response.read().decode())
        
    with open("/Users/sanjaysinghrawat/Desktop/scaler-ai-persona/data/github_context.txt", "w") as f:
        f.write("GitHub User: sanjayrawatt\n\n")
        for repo in repos:
            if not repo['fork']:
                f.write(f"Repo: {repo['name']}\n")
                f.write(f"Description: {repo['description']}\n")
                f.write(f"Tech Stack/Language: {repo['language']}\n")
                f.write(f"URL: {repo['html_url']}\n\n")
    print("Repos fetched successfully.")
except Exception as e:
    print(f"Error fetching repos: {e}")
