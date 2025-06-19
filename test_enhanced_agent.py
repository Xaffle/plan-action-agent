"""
增强智能体测试文件

测试新的智能体功能：
1. LLM驱动的控制流
2. 自我反思机制
3. 动态计划调整
"""

from enhanced_agents import run_enhanced_agent

def test_simple_objective():
    """测试简单目标"""
    print("🧪 测试1: 简单学习计划")
    objective = "制定一个今天学习Python基础的计划，并模拟执行第一步"
    
    result = run_enhanced_agent(objective, max_iterations=8)
    
    print("\n📊 测试结果:")
    print(f"完成任务数: {len(result['completed_tasks'])}")
    print(f"反思次数: {len(result['reflections'])}")
    print(f"最终信心: {result['final_confidence']:.2f}")
    print(f"状态: {result['status']}")
    
    return result

def test_complex_objective():
    """测试复杂目标"""
    print("\n" + "="*60)
    print("🧪 测试2: 复杂项目规划")
    objective = "为我的团队规划一个为期一周的网站开发项目，包括需求分析、技术选型、开发分工和时间安排"
    
    result = run_enhanced_agent(objective, max_iterations=12)
    
    print("\n📊 测试结果:")
    print(f"完成任务数: {len(result['completed_tasks'])}")
    print(f"反思次数: {len(result['reflections'])}")
    print(f"最终信心: {result['final_confidence']:.2f}")
    print(f"状态: {result['status']}")
    
    return result

def test_adaptive_behavior():
    """测试自适应行为"""
    print("\n" + "="*60)
    print("🧪 测试3: 自适应问题解决")
    objective = "帮我解决一个编程难题：如何设计一个高效的缓存系统，要求考虑并发安全、内存管理和性能优化"
    
    result = run_enhanced_agent(objective, max_iterations=15)
    
    print("\n📊 测试结果:")
    print(f"完成任务数: {len(result['completed_tasks'])}")
    print(f"反思次数: {len(result['reflections'])}")
    print(f"最终信心: {result['final_confidence']:.2f}")
    print(f"状态: {result['status']}")
    
    # 打印一些详细结果
    if result['completed_tasks']:
        print("\n🔍 完成的任务:")
        for i, task in enumerate(result['completed_tasks'][:3], 1):
            print(f"{i}. {task['task']}")
            print(f"   质量分数: {task.get('quality_score', 'N/A')}")
    
    if result['reflections']:
        print("\n🤔 反思记录:")
        for i, reflection in enumerate(result['reflections'][:2], 1):
            print(f"{i}. {reflection[:150]}...")
    
    return result

if __name__ == "__main__":
    print("🚀 开始测试增强智能体...")
    
    try:
        # 运行测试
        test1_result = test_simple_objective()
        test2_result = test_complex_objective()
        test3_result = test_adaptive_behavior()
        
        print("\n" + "="*60)
        print("🎉 所有测试完成！")
        print("\n📈 测试总结:")
        print(f"测试1 - 简单目标: {test1_result['status']}")
        print(f"测试2 - 复杂目标: {test2_result['status']}")
        print(f"测试3 - 自适应行为: {test3_result['status']}")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
