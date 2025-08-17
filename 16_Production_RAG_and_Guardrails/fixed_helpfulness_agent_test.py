#!/usr/bin/env python3
"""
Fixed Helpfulness Agent Test - Demonstrates corrected evaluation and refinement logic
"""

import time
from langchain_core.messages import HumanMessage
from langgraph_agent_lib import create_helpfulness_agent, create_langgraph_agent, get_openai_model

def helpfulness_evaluator(inputs: dict, outputs: dict) -> dict:
    """LangSmith evaluator for helpfulness assessment - returns raw scores."""
    try:
        prompt = f"""You are an expert evaluator assessing the helpfulness of AI responses.

User's question: {inputs.get('question', '')}
AI response: {outputs.get('response', '')}

Evaluate the response based on these criteria:
1. Relevance: Does it directly address the user's question?
2. Completeness: Is the information comprehensive and complete?
3. Clarity: Is it clear and easy to understand?
4. Actionability: Does it provide useful, actionable guidance when appropriate?
5. Accuracy: Is the information correct and well-sourced?

Rate the overall helpfulness on a scale of 1-10 (1=not helpful at all, 10=extremely helpful).

Provide ONLY your numerical rating as: SCORE: X"""
        
        eval_model = get_openai_model(model_name="gpt-4.1-mini", temperature=0.0)
        response = eval_model.invoke(prompt)
        
        content = response.content.upper()
        score = 5.0
        if "SCORE:" in content:
            try:
                score_part = content.split("SCORE:")[1].strip()
                if "/" in score_part:
                    score_part = score_part.split("/")[0]
                score = float(score_part)
                score = max(1.0, min(10.0, score))
            except:
                score = 5.0
        
        return {
            "helpfulness_score": score,
            "evaluation_reasoning": response.content
        }
    except Exception as e:
        return {
            "helpfulness_score": 5.0,
            "evaluation_reasoning": f"Evaluation failed: {str(e)}"
        }

def analyze_agent_conversation(response):
    """Analyze agent conversation to detect actual refinements."""
    messages = response["messages"]
    
    # Find helpfulness evaluations
    helpfulness_evals = [m for m in messages if hasattr(m, 'content') and 'HELPFULNESS:' in str(m.content)]
    
    # Find actual agent responses (exclude initial user message and helpfulness messages)
    agent_responses = []
    for msg in messages:
        if (hasattr(msg, 'content') and 
            not str(msg.content).startswith('HELPFULNESS:') and
            not getattr(msg, 'tool_calls', None) and
            msg != messages[0]):  # Exclude initial user message
            agent_responses.append(msg)
    
    # Check for refinement instructions
    refinement_instructions = [m for m in messages if hasattr(m, 'content') and 'previous response scored' in str(m.content).lower()]
    
    return {
        'total_messages': len(messages),
        'helpfulness_evaluations': len(helpfulness_evals),
        'agent_responses': len(agent_responses),
        'refinement_instructions': len(refinement_instructions),
        'was_actually_refined': len(agent_responses) > 1,
        'evaluation_scores': [eval(msg.content.split('SCORE:')[1]) if 'SCORE:' in msg.content else None for msg in helpfulness_evals]
    }

def test_improved_helpfulness_agent():
    """Test the improved helpfulness agent with better detection logic."""
    print("🔧 TESTING IMPROVED HELPFULNESS AGENT")
    print("=" * 60)
    
    # Create agents for comparison
    try:
        from langgraph_agent_lib import ProductionRAGChain
        rag_chain = ProductionRAGChain(
            file_path="./data/The_Direct_Loan_Program.pdf",
            chunk_size=1000,
            chunk_overlap=100,
            embedding_model="text-embedding-3-small",
            llm_model="gpt-4.1-mini",
            cache_dir="./cache"
        )
    except Exception as e:
        print(f"⚠ Could not create RAG chain: {e}")
        rag_chain = None
    
    simple_agent = create_langgraph_agent(
        model_name="gpt-4.1-mini",
        temperature=0.1,
        rag_chain=rag_chain
    )
    
    helpfulness_agent = create_helpfulness_agent(
        model_name="gpt-4.1-mini",
        temperature=0.1,
        rag_chain=rag_chain,
        max_loops=2,
        helpfulness_threshold=7.0  # Set threshold to trigger refinements
    )
    
    # Test queries that should trigger refinements
    test_queries = [
        "What is student loan forgiveness?",  # Simple query - should score well
        "Tell me about loans.",  # Vague query - should trigger refinement
        "How do I get financial help?",  # Broad query - might trigger refinement
    ]
    
    results = []
    
    for i, query in enumerate(test_queries):
        print(f"\n🔍 Test {i+1}: {query}")
        print("-" * 50)
        
        # Test Simple Agent
        print("📝 Simple Agent:")
        start_time = time.time()
        simple_response = simple_agent.invoke({"messages": [HumanMessage(content=query)]})
        simple_time = time.time() - start_time
        simple_analysis = analyze_agent_conversation(simple_response)
        
        # Get final response
        simple_final = None
        for msg in reversed(simple_response["messages"]):
            if (hasattr(msg, 'content') and 
                not str(msg.content).startswith('HELPFULNESS:') and
                not getattr(msg, 'tool_calls', None)):
                simple_final = msg
                break
        
        simple_eval = helpfulness_evaluator(
            inputs={"question": query},
            outputs={"response": simple_final.content if simple_final else "No response"}
        )
        
        print(f"  ⏱ Time: {simple_time:.2f}s")
        print(f"  📊 Score: {simple_eval['helpfulness_score']}/10")
        print(f"  🔄 Messages: {simple_analysis['total_messages']}")
        print(f"  ✅ Refined: {simple_analysis['was_actually_refined']}")
        
        # Test Helpfulness Agent
        print("\n🤖 Helpfulness Agent:")
        start_time = time.time()
        helpful_response = helpfulness_agent.invoke({"messages": [HumanMessage(content=query)]})
        helpful_time = time.time() - start_time
        helpful_analysis = analyze_agent_conversation(helpful_response)
        
        # Get final response
        helpful_final = None
        for msg in reversed(helpful_response["messages"]):
            if (hasattr(msg, 'content') and 
                not str(msg.content).startswith('HELPFULNESS:') and
                not getattr(msg, 'tool_calls', None)):
                helpful_final = msg
                break
        
        helpful_eval = helpfulness_evaluator(
            inputs={"question": query},
            outputs={"response": helpful_final.content if helpful_final else "No response"}
        )
        
        print(f"  ⏱ Time: {helpful_time:.2f}s")
        print(f"  📊 Score: {helpful_eval['helpfulness_score']}/10")
        print(f"  🔄 Messages: {helpful_analysis['total_messages']}")
        print(f"  ✅ Refined: {helpful_analysis['was_actually_refined']}")
        print(f"  🔍 Evaluations: {helpful_analysis['helpfulness_evaluations']}")
        print(f"  📝 Refinement Instructions: {helpful_analysis['refinement_instructions']}")
        
        if helpful_analysis['evaluation_scores']:
            print(f"  📈 Internal Scores: {helpful_analysis['evaluation_scores']}")
        
        # Compare results
        score_improvement = helpful_eval['helpfulness_score'] - simple_eval['helpfulness_score']
        time_overhead = helpful_time - simple_time
        
        print(f"\n📈 COMPARISON:")
        print(f"  🎯 Score Improvement: {score_improvement:+.1f} points")
        print(f"  ⏱ Time Overhead: {time_overhead:+.2f}s")
        print(f"  🤔 Actually Refined: {helpful_analysis['was_actually_refined']}")
        
        results.append({
            'query': query,
            'simple_score': simple_eval['helpfulness_score'],
            'helpful_score': helpful_eval['helpfulness_score'],
            'score_improvement': score_improvement,
            'simple_time': simple_time,
            'helpful_time': helpful_time,
            'time_overhead': time_overhead,
            'actually_refined': helpful_analysis['was_actually_refined'],
            'internal_evaluations': helpful_analysis['helpfulness_evaluations']
        })
    
    # Summary
    print(f"\n🏆 SUMMARY RESULTS")
    print("=" * 50)
    avg_improvement = sum(r['score_improvement'] for r in results) / len(results)
    avg_overhead = sum(r['time_overhead'] for r in results) / len(results)
    refinement_rate = sum(1 for r in results if r['actually_refined']) / len(results) * 100
    
    print(f"📊 Average Score Improvement: {avg_improvement:+.1f} points")
    print(f"⏱ Average Time Overhead: {avg_overhead:+.2f}s")
    print(f"🔄 Refinement Rate: {refinement_rate:.1f}%")
    print(f"🎯 Tests with Actual Refinement: {sum(1 for r in results if r['actually_refined'])}/{len(results)}")
    
    return results

if __name__ == "__main__":
    test_improved_helpfulness_agent()
