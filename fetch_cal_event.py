import urllib.request
import json
import os

cal_key = "cal_live_bf3c21c230e4aa9b2495ed2492d5344e"
url = f"https://api.cal.com/v1/event-types?apiKey={cal_key}"
req = urllib.request.Request(url)
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        event_types = data.get("event_types", [])
        if event_types:
            event_id = event_types[0]["id"]
            print(f"Found event type ID: {event_id}")
            
            # Update .env file
            env_path = "/Users/sanjaysinghrawat/Desktop/scaler-ai-persona/.env"
            with open(env_path, "r") as f:
                content = f.read()
            content = content.replace('CAL_EVENT_TYPE_ID=""', f'CAL_EVENT_TYPE_ID="{event_id}"')
            with open(env_path, "w") as f:
                f.write(content)
            print("Successfully updated .env with CAL_EVENT_TYPE_ID")
        else:
            print("No event types found on Cal.com account.")
except Exception as e:
    print(f"Error: {e}")
