import json
import requests

# Load the caseTable.json file
with open('caseTable.json', 'r') as file:
    case_table = json.load(file)

# Set the base URL and authorization token
base_url = "https://www.courtlistener.com/api/rest/v3/docket-alerts/"
auth_token = "a90d3f2de489aa4138a32133ca8bfec9d85fecfa"

# Iterate through each key (docket ID) in the case table
for docket_id in case_table.keys():
    # Set the data payload and headers for the request
    data = {'docket': docket_id}
    headers = {'Authorization': f'Token {auth_token}'}

    try:
        # Send the POST request to the CourtListener API
        response = requests.post(base_url, data=data, headers=headers)

        # Check the response status code
        if response.status_code == 200:
            print(f"Successfully created docket alert for docket ID: {docket_id}")
        else:
            print(f"Failed to create docket alert for docket ID: {docket_id}")
            print(f"Status code: {response.status_code}")
            print(f"Response content: {response.content}")

    except requests.exceptions.RequestException as e:
        print(f"Error occurred while creating docket alert for docket ID: {docket_id}")
        print(f"Error message: {str(e)}")
