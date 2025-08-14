# 🎬 Expert Agent Demo Setup Guide

---

## 📑 Table of Contents

- [🎯 What You'll Demonstrate](#what-youll-demonstrate)
- [🖥️ Terminal Setup](#️-terminal-setup)
- [🚀 Step-by-Step Instructions](#-step-by-step-instructions)
- [📋 Demo Scenarios Guide](#-demo-scenarios-guide)
- [🎭 Advanced Demo Flow](#-advanced-demo-flow)
- [🎯 Key Demo Points to Highlight](#-key-demo-points-to-highlight)
- [🎪 Advanced Demo Techniques](#-advanced-demo-techniques)

**Navigation**: [🏠 Main README](./README.md) | [📋 Assignment Answers](./ASSIGNMENT_ANSWERS.md) | [🤖 Expert Agent](./second_agent/README.md) | [🏭 App Docs](./app/README.md)

---

This guide shows you how to run impressive demos of our **Goal-Oriented Expert Agent System** with comprehensive side-by-side terminal logging.

## 🎯 What You'll Demonstrate

Our system now features **4 specialized expert agents** with specific research missions:

1. **🔬 Dr. Sarah Chen** - ML Expert studying Kimi K2 (Assignment Example)
2. **🧠 Prof. Marcus Rodriguez** - Transformer Architecture Researcher  
3. **💼 Alex Kim** - AI Startup Founder
4. **🔒 Dr. Emma Watson** - AI Security Expert

Each expert has **quality standards** and will ask **follow-up questions** if unsatisfied!

## 🖥️ Terminal Setup

You'll need **TWO terminal windows** side by side:

```
┌─────────────────────────────────┬─────────────────────────────────┐
│         LEFT TERMINAL           │         RIGHT TERMINAL          │
│      Main Agent Server          │      Expert Agent System       │
│    (Receives requests)          │     (Goal-driven experts)       │
│                                 │                                 │
│  🏭 MAIN AGENT SERVER           │  👨‍🔬 EXPERT AGENT SYSTEM      │
│  📊 Processing expert query...  │  🎯 Dr. Sarah Chen studying... │
│  🛠️ Executing ArXiv search...   │  📊 Quality evaluation: 6/10   │
│  ✅ Response generated          │  🔄 Follow-up needed...         │
│                                 │  📞 Calling again...            │
└─────────────────────────────────┴─────────────────────────────────┘
```

## 🚀 Step-by-Step Instructions

### Terminal 1 (Left): Start Main Agent Server

```bash
# Navigate to project directory
cd /Users/ovookpubuluku/project-repos/ai-makerspace/AIE7/15_A2A_LangGraph

# Start the main agent with enhanced logging
uv run python start_main_agent_with_logging.py
```

**You should see:**
```
🏭 MAIN AGENT SERVER - ENHANCED LOGGING MODE
============================================================
🎯 This is the main A2A agent that the second agent will call
📊 All processing steps will be logged in detail
🔄 Waiting for requests from the second agent...
============================================================
INFO     | MainAgent | INFO | 🚀 Enhanced logging enabled for A2A agent server
INFO     | MainAgent | INFO | 🎯 Ready to receive requests from second agent
INFO     | MainAgent | INFO | 📊 All agent interactions will be logged below
```

### Terminal 2 (Right): Start Expert Agent System

```bash
# Navigate to project directory (in a new terminal)
cd /Users/ovookpubuluku/project-repos/ai-makerspace/AIE7/15_A2A_LangGraph

# Start the expert agent system
uv run python second_agent/main.py
```

**You'll see:**
```
🎓 EXPERT AGENT SYSTEM - Advanced A2A Communication
============================================================
🎯 Goal-oriented experts with specific research missions!
📊 Featuring: Dr. Sarah Chen studying Kimi K2 (Assignment Example)
✅ A2A agent server is running

Choose demo type:
1. Single Query Demo (quick test)
2. Multi-Persona Demo (original system)
3. 🌟 Expert Agent Mode (NEW - Assignment Compliant)

Enter choice (1, 2, or 3): 
```

## 📋 Demo Scenarios Guide

### 🎬 Scenario 1: Assignment Example Demo (Recommended)

**Purpose**: Demonstrate exact assignment requirements with Dr. Sarah Chen

**Steps**:
1. **Choose option 3** (Expert Agent Mode)
2. **Select expert 1** (Dr. Sarah Chen - ML Expert studying Kimi K2)
3. **Ask the key question**: `What makes Kimi K2 so incredible?`

**What to highlight**:
- Watch expert's **quality evaluation** (likely will score low initially)
- Observe **follow-up questions** when not satisfied
- Show **persistent goal** throughout conversation
- Point out **sources requirement** being enforced

**Expected flow**:
```
🔬 Dr. Sarah Chen asks: What makes Kimi K2 so incredible?
→ Expert evaluates response: 5/10 (not enough sources)
→ Follow-up: "I need sources and references to verify this information..."
→ Main agent provides ArXiv papers and technical details
→ Expert satisfaction improves: 8/10
```

### 🎬 Scenario 2: Transformer Research Expert

**Purpose**: Show academic research focus with Prof. Marcus Rodriguez

**Steps**:
1. **Choose option 3** (Expert Agent Mode)
2. **Select expert 2** (Prof. Marcus Rodriguez)
3. **Ask**: `What are the latest innovations in attention mechanisms?`

**What to highlight**:
- Expert demands **academic rigor and citations**
- Follow-up strategy: **mathematical explanations and code examples**
- Watch for requests for **technical implementation details**

### 🎬 Scenario 3: Business Analysis Expert

**Purpose**: Demonstrate business-focused AI evaluation with Alex Kim

**Steps**:
1. **Choose option 3** (Expert Agent Mode)  
2. **Select expert 3** (Alex Kim - Startup Founder)
3. **Ask**: `Should I build my AI product on LangChain or LlamaIndex?`

**What to highlight**:
- Expert focuses on **practical implementation** and **ROI**
- Follow-up strategy: **real-world examples and scalability**
- Quality standards: **cost analysis and business metrics**

### 🎬 Scenario 4: Security Expert Analysis

**Purpose**: Show cybersecurity focus with Dr. Emma Watson

**Steps**:
1. **Choose option 3** (Expert Agent Mode)
2. **Select expert 4** (Dr. Emma Watson - Security Expert)  
3. **Ask**: `What are the security risks in large language models?`

**What to highlight**:
- Expert demands **concrete vulnerability examples**
- Follow-up strategy: **specific attack vectors and defenses**
- Quality standards: **evidence-based mitigation strategies**

### 🎬 Scenario 5: Expert Switching Demo

**Purpose**: Show multiple experts in one session

**Steps**:
1. Start with any expert
2. **Type**: `switch`
3. **Select different expert**
4. **Ask related question**

**What to highlight**:
- Different experts have **different approaches** to similar topics
- **Persistent goals** and **quality standards** per expert
- **Context switching** maintains expert identity

## 🎭 Advanced Demo Flow

### Multi-Turn Expert Conversations

**Dr. Sarah Chen Sequence**:
```
🔬 Dr. Sarah Chen asks: What makes Kimi K2 so incredible?
→ (Gets general response)
→ Expert: "This seems surface-level. I need technical details..."
→ (Gets more technical response with sources)
→ Expert: "Better! Can you provide specific papers to verify?"
→ (Gets ArXiv papers and citations)
→ Expert satisfaction: 9/10 ✅
```

**Alex Kim Business Sequence**:
```
💼 Alex Kim asks: What are the costs of deploying LLMs?
→ (Gets technical response)
→ Expert: "I need practical pricing and ROI data..."
→ (Gets business-focused cost analysis)
→ Expert satisfaction: 8/10 ✅
```

## 📊 What You'll See

### Terminal 1 (Main Agent Server) Logs:
```
================================================================================
📥 NEW REQUEST FROM EXPERT AGENT
================================================================================
📝 Query: As an expert in Machine Learning, I am working on: learn about what makes Kimi K2 so incredible
My question: What makes Kimi K2 so incredible?
Please provide a response that meets my standards: not satisfied with surface level answers, want sources to read to verify information
🎭 Expert context detected: You are an expert in Machine Learning seeking detailed technical information...
🔄 Starting processing...
🛠️  Executing ArxivQueryRun  
📊 Tool input: Kimi K2 machine learning model
✅ Response generated successfully
📝 Response length: 892 characters
⏱️  Processing time: 4.12 seconds
👀 Response preview: Kimi K2 is a large language model developed by Moonshot AI...
📤 Sending response back to expert agent
================================================================================
```

### Terminal 2 (Expert Agent) Logs:
```
👨‍🔬 EXPERT SELECTION: Determining expert profile...
🎭 Continuing with expert: Dr. Sarah Chen
🧠 Expert Identity: an expert in Machine Learning
🎯 Current Goal: learn about what makes Kimi K2 so incredible
📊 Quality Standards: ['not satisfied with surface level answers', 'want sources to read to verify information']
🔄 Follow-up Strategy: If initial answer lacks depth, ask for technical details, papers, or implementation specifics
📞 A2A CALL NODE: Preparing expert consultation...
🎯 Expert: Dr. Sarah Chen (an expert in Machine Learning)
🎯 Goal: learn about what makes Kimi K2 so incredible
🎭 Using expert persona context (425 chars)
📝 Expert query: As an expert in Machine Learning, I am working on...
🚀 Consulting main agent as expert...
📥 Received response from main agent (4.12s)
📊 Parsing response from main agent...
✅ Successfully parsed response from main agent
🔍 QUALITY EVALUATION: Assessing response quality...
📊 Quality Score: 6/10
🎯 Expert Standards: ['not satisfied with surface level answers', 'want sources to read to verify information']
❌ Response quality insufficient - follow-up needed
🔄 Follow-up strategy: If initial answer lacks depth, ask for technical details, papers, or implementation specifics
🔄 FOLLOW-UP GENERATION: Creating expert follow-up...
🎯 Follow-up type: sources
❓ Follow-up question: I need sources and references to verify this information...
```

## 🎯 Key Demo Points to Highlight

### 🌟 **Assignment Compliance**
1. **✅ Different Agent Framework**: LangGraph expert system vs A2A server
2. **✅ Specific Expert Personas**: Dr. Sarah Chen studying Kimi K2 (exact assignment example)
3. **✅ Goal-Oriented Behavior**: Persistent research missions with quality standards
4. **✅ A2A Communication**: Full protocol compliance with enhanced context

### 🔧 **Advanced Features**
5. **📊 Quality Evaluation**: Real-time scoring of response quality (0-10 scale)
6. **🔄 Follow-up Logic**: Automatic generation of follow-up questions when unsatisfied
7. **🎭 Expert Persistence**: Maintains identity and standards across conversation
8. **📞 Enhanced A2A Protocol**: Expert context added to all communications

### 🚀 **Technical Highlights**
9. **⏱️ Performance Metrics**: Real-time timing and satisfaction scoring
10. **🛠️ Tool Orchestration**: Main agent uses appropriate tools based on expert context
11. **✨ Multi-Turn Conversations**: Experts continue until satisfaction achieved
12. **🔍 Comprehensive Logging**: Full visibility into expert decision-making

## 💡 Demo Script Suggestions

### 🎬 **Opening Script**:
*"Today I'll show you our Goal-Oriented Expert Agent System. Unlike simple chatbots, these are AI experts with specific research missions, quality standards, and the persistence to keep asking until they get the answers they need."*

### 🎯 **Assignment Example Script**:
*"Let me demonstrate the exact assignment example: Dr. Sarah Chen, an ML expert studying Kimi K2. Watch how she's not satisfied with surface-level answers and demands sources to verify information."*

### 🔄 **Follow-up Demo Script**:
*"Notice how Dr. Chen evaluated that response as 6/10 - not good enough for her standards. She's now generating a follow-up question asking for sources and references. This shows true expert behavior!"*

### 🎭 **Expert Switching Script**:
*"Now let me show you Alex Kim, our startup founder. Same topic, but watch how his business perspective leads to completely different questions about costs and ROI rather than academic papers."*

## 🎪 Advanced Demo Techniques

### 🖥️ **Terminal Setup**:
1. **Use iTerm2 split panes** for seamless side-by-side view
2. **Different color schemes** (dark vs light) for visual distinction
3. **Larger font size** for audience visibility
4. **Recording setup** with OBS for later review

### 🎯 **Demo Flow**:
1. **Start with assignment example** (Dr. Sarah Chen + Kimi K2)
2. **Show follow-up behavior** when unsatisfied
3. **Switch experts** to demonstrate different approaches
4. **Highlight quality scoring** and expert persistence
5. **End with technical deep-dive** into A2A protocol logs

### 📊 **Audience Engagement**:
- **Ask audience for questions** to test different experts
- **Predict expert behavior** before running queries
- **Compare expert responses** to the same question
- **Explain quality evaluation** scoring system

This demonstrates a **production-ready expert agent system** that goes far beyond simple persona switching - it shows true goal-oriented AI behavior! 🎉

