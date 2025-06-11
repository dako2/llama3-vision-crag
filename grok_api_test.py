import requests
import json

# Define API endpoint and your API key
API_URL ="https://api.x.ai/v1"
API_KEY = "xai-s3R1gXkvA9se2FBbQch3TowVZk0y7X1DPpQ9GDjh9FK5Lhs5JUY8AOxThXHSfhtnlOIl2EBjB5IqKeOr"  # Replace with your actual API key

# Sample function to test Grok API
def test_grok_api(query="Hello, tell me about the universe!"):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Prepare the payload
    payload = {
        "query": query,
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        # Make the API request
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse and print the response
        result = response.json()
        print("Grok API Response:")
        print(json.dumps(result, indent=2))
        
        # Extract and return the response text if available
        if "response" in result:
            return result["response"]
        else:
            return "No response field in API result."
            
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return None

# Test the function
if __name__ == "__main__":
    test_query = "Explain the theory of relativity in simple terms."
    response_text = test_grok_api(test_query)
    if response_text:
        print("\nResponse Text:")
        print(response_text)
