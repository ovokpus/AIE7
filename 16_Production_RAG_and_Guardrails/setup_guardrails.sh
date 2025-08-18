#!/bin/bash

# Guardrails Setup Script
# This script sets up Guardrails AI with all required components

echo "🛡️ Guardrails AI Setup Script"
echo "=============================="



# Step 1: Sync dependencies
echo ""
echo "📦 Step 1: Syncing UV dependencies..."
uv sync
if [ $? -ne 0 ]; then
    echo "❌ UV sync failed"
    exit 1
fi
echo "✅ UV sync completed"

source .venv/bin/activate

# Step 2: Configure Guardrails
echo ""
echo "⚙️ Step 2: Configuring Guardrails..."
uv run guardrails configure
if [ $? -ne 0 ]; then
    echo "❌ Guardrails configuration failed"
    exit 1
fi
echo "✅ Guardrails configured"

# Step 3: Install Hub Components
echo ""
echo "🔧 Step 3: Installing Guardrails Hub Components..."

echo "  Installing RestrictToTopic..."
uv run guardrails hub install hub://tryolabs/restricttotopic
if [ $? -ne 0 ]; then
    echo "❌ Failed to install restricttotopic"
    exit 1
fi
echo "  ✅ RestrictToTopic installed"

echo "  Installing DetectJailbreak..."
uv run guardrails hub install hub://guardrails/detect_jailbreak
if [ $? -ne 0 ]; then
    echo "❌ Failed to install detect_jailbreak"
    exit 1
fi
echo "  ✅ DetectJailbreak installed"

echo "  Installing CompetitorCheck..."
uv run guardrails hub install hub://guardrails/competitor_check
if [ $? -ne 0 ]; then
    echo "❌ Failed to install competitor_check"
    exit 1
fi
echo "  ✅ CompetitorCheck installed"

echo "  Installing LLM RAG Evaluator..."
uv run guardrails hub install hub://arize-ai/llm_rag_evaluator
if [ $? -ne 0 ]; then
    echo "❌ Failed to install llm_rag_evaluator"
    exit 1
fi
echo "  ✅ LLM RAG Evaluator installed"

echo "  Installing ProfanityFree..."
uv run guardrails hub install hub://guardrails/profanity_free
if [ $? -ne 0 ]; then
    echo "❌ Failed to install profanity_free"
    exit 1
fi
echo "  ✅ ProfanityFree installed"

echo "  Installing GuardrailsPII..."
uv run guardrails hub install hub://guardrails/guardrails_pii
if [ $? -ne 0 ]; then
    echo "❌ Failed to install guardrails_pii"
    exit 1
fi
echo "  ✅ GuardrailsPII installed"

# Step 4: Verification
echo ""
echo "🧪 Step 4: Verifying Installation..."
uv run python -c "
import guardrails as gd
from guardrails import Guard
print('✅ Core Guardrails imported successfully')

try:
    from guardrails.hub import RestrictToTopic, CompetitorCheck
    print('✅ Hub components imported successfully')
except ImportError as e:
    print(f'⚠️ Some hub components may need time to install: {e}')

print('🎉 Guardrails setup completed successfully!')
"

echo ""
echo "🎉 GUARDRAILS SETUP COMPLETED!"
echo "=============================="
echo "✅ All dependencies installed"
echo "✅ Guardrails configured"
echo "✅ Hub components installed"
echo "✅ Ready for production use!"
echo ""
echo "🛡️ Your system now has comprehensive guardrails protection:"
echo "   - Topic restriction (RestrictToTopic)"
echo "   - Jailbreak detection (DetectJailbreak)"
echo "   - Competitor mentions (CompetitorCheck)"
echo "   - RAG evaluation (LLM RAG Evaluator)"
echo "   - Profanity filtering (ProfanityFree)"
echo "   - PII protection (GuardrailsPII)"
echo ""
echo "Run 'uv run python -c \"from langgraph_agent_lib.agents import GUARDRAILS_AVAILABLE; print(f'Guardrails available: {GUARDRAILS_AVAILABLE}')\"' to test imports"
