#!/bin/bash

# Iterate from 18 to 30
for i in $(seq -w 01 31); do
  # Construct the date string
  DATE="2024-07-${i}"

  # Print the date being processed (optional)
  echo "Processing date: $DATE"

  # Run the curl command
  curl -X POST -H "Content-Type: application/json" -d '{"mood": "joyful"}' "http://localhost:4444/note/banner?dt=$DATE"

  # Wait for the curl command to finish before starting the next iteration
  wait
done

