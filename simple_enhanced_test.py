#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enhanced_agents import run_enhanced_agent

def test_simple_objective():
    """测试简单目标"""
    print("🧪 测试1: 简单学习计划")
    objective = "制定一个今天学习Python基础的计划，并模拟执行第一步"
    
    result = run_enhanced_agent(objective, max_iterations=6)
    
    print("\n📊 测试结果:")
    print(f"完成任务数: {len(result['completed_tasks'])}")
    print(f"反思次数: {len(result['reflections'])}")
    print(f"最终信心: {result['final_confidence']:.2f}")
    print(f"状态: {result['status']}")
    
    return result

if __name__ == "__main__":
    print("🚀 开始测试增强智能体...")
    
    try:
        # 运行测试
        test1_result = test_simple_objective()
        
        print("\n" + "="*60)
        print("🎉 测试完成！")
        print(f"状态: {test1_result['status']}")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
