
#!/bin/bash

# Script to monitor continuous_trellis.log for occurrences of '0 prompts'

LOG_FILE="continuous_trellis.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file '$LOG_FILE' not found."
    exit 1
fi

echo "Monitoring $LOG_FILE for lines containing '0 prompts'..."

tail -f "$LOG_FILE" | grep --line-buffered "0 prompts" 