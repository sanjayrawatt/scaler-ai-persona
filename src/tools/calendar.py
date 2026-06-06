import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

CAL_API_KEY = os.environ.get("CAL_API_KEY")
EVENT_TYPE_ID = os.environ.get("CAL_EVENT_TYPE_ID")

def get_availability(date_from: str, date_to: str):
    """Fetch available slots from Cal.com using v2 and return them in IST"""
    if not CAL_API_KEY:
        return "Cal.com API key missing"
    
    url = f"https://api.cal.com/v2/slots/available?eventTypeId={EVENT_TYPE_ID}&startTime={date_from}&endTime={date_to}"
    headers = {
        "Authorization": f"Bearer {CAL_API_KEY}",
        "cal-api-version": "2024-08-13"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # Convert UTC slots to IST for the AI to read
        converted = {}
        if "data" in data and "slots" in data["data"]:
            for date_key, slots in data["data"]["slots"].items():
                ist_slots = []
                for slot in slots:
                    utc_time = datetime.fromisoformat(slot["time"].replace("Z", "+00:00"))
                    ist_time = utc_time + timedelta(hours=5, minutes=30)
                    ist_slots.append({
                        "time_ist": ist_time.strftime("%I:%M %p IST"),
                        "time_utc": slot["time"]
                    })
                converted[date_key] = ist_slots
        return {"status": "success", "available_slots": converted, "note": "When booking, use the time_utc value for start_time parameter."}
    return {"status": "error", "message": f"Cal.com API error: {response.text}"}

def book_meeting(name: str, email: str, start_time: str, timezone: str = "Asia/Kolkata"):
    """Book a meeting on Cal.com using v2"""
    if not CAL_API_KEY:
        return "Cal.com API key missing"
    
    # If the start_time doesn't end with Z or contain +/-, it might be IST
    # Convert IST to UTC by subtracting 5:30
    if "Z" not in start_time and "+" not in start_time and "-" not in start_time[10:]:
        try:
            local_time = datetime.fromisoformat(start_time)
            utc_time = local_time - timedelta(hours=5, minutes=30)
            start_time = utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"Converted IST to UTC: {start_time}")
        except Exception as e:
            print(f"Time conversion error: {e}")
        
    url = "https://api.cal.com/v2/bookings"
    headers = {
        "Authorization": f"Bearer {CAL_API_KEY}",
        "cal-api-version": "2024-08-13",
        "Content-Type": "application/json"
    }
    
    payload = {
        "start": start_time,
        "eventTypeId": int(EVENT_TYPE_ID),
        "attendee": {
            "name": name,
            "email": email,
            "timeZone": timezone
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code in [200, 201]:
        return f"Meeting booked successfully! Confirmation sent to {email}."
    
    return f"Booking failed: {response.text}. Please try a different time slot."

