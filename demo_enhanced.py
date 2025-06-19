#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enhanced_agents import run_enhanced_agent
import json

def test_enhanced_agent():
    """测试增强智能体的实际功能"""
    print("=== Enhanced Agent Demo ===")
    print("Testing LLM-driven control flow, self-reflection, and tool usage\n")
    
    # 测试目标
    objective = "Help me create a simple study plan for learning Python web development with Flask, and simulate the first learning session"
    
    print(f"Objective: {objective}")
    print("\nStarting enhanced agent...\n")
    print("=" * 60)
    
    try:
        # 运行增强智能体
        result = run_enhanced_agent(objective, max_iterations=8)
        
        print("\n" + "=" * 60)
        print("=== EXECUTION SUMMARY ===")
        print(f"Status: {result['status']}")
        print(f"Iterations used: {result['iterations_used']}")
        print(f"Tasks completed: {len(result['completed_tasks'])}")
        print(f"Reflections made: {len(result['reflections'])}")
        print(f"Final confidence: {result['final_confidence']:.2f}")
        
        if result['completed_tasks']:
            print("\n=== COMPLETED TASKS ===")
            for i, task in enumerate(result['completed_tasks'], 1):
                print(f"\n{i}. Task: {task['task']}")
                print(f"   Quality Score: {task.get('quality_score', 'N/A')}")
                if 'results' in task:
                    print(f"   Results: {task['results'][:150]}...")
        
        if result['reflections']:
            print("\n=== REFLECTIONS ===")
            for i, reflection in enumerate(result['reflections'], 1):
                try:
                    refl_data = json.loads(reflection)
                    print(f"\n{i}. Assessment: {refl_data.get('assessment', 'N/A')[:100]}...")
                    if refl_data.get('recommendations'):
                        print(f"   Recommendations: {refl_data['recommendations'][:2]}")
                except:
                    print(f"\n{i}. Raw reflection: {reflection[:100]}...")
        
        print("\n=== DEMO COMPLETED ===")
        return result
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_enhanced_agent()
