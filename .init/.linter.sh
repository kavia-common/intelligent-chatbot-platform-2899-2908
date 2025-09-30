#!/bin/bash
cd /home/kavia/workspace/code-generation/intelligent-chatbot-platform-2899-2908/company_chatbot_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

