"""
智能体测试脚本 - 简单版本

这个脚本用于测试智能体的基本功能：
1. 设置测试环境（API密钥检查）
2. 定义测试任务
3. 运行智能体
4. 展示执行结果

测试用例说明：
- 选择了一个中等复杂度的任务（制定学习计划）
- 既需要规划能力（分解学习任务）
- 也需要执行能力（具体安排时间和内容）
"""

import os
from agents import run_agent
from api_setting import API_KEY

# ==================== 测试任务定义 ====================
# 选择一个适中复杂度的任务来测试智能体能力
simple_objective = "帮我制定一个简单的周末学习计划，包括编程练习和阅读"

print("🚀 开始运行智能体...")
print(f"🎯 目标: {simple_objective}")
print("=" * 60)

# ==================== 执行智能体 ====================
try:
    # 调用智能体执行任务
    results = run_agent(simple_objective)
    
    # ==================== 结果展示 ====================
    print("\n🎉 执行完成！")
    print("=" * 60)
    print("📊 最终结果统计:")
    print(f"   - 总共执行了 {len(results)} 个步骤")
    print("=" * 60)
    print("📋 详细执行结果:")
    
    # 逐个展示每个任务的执行结果
    for i, result in enumerate(results, 1):
        print(f"\n🔸 步骤 {i} 详细结果:")
        print("=" * 40)
        print(result)
        print("=" * 40)
        
except Exception as e:
    # ==================== 错误处理 ====================
    print(f"❌ 执行出错: {e}")
    import traceback
    print("\n🔍 详细错误信息:")
    traceback.print_exc()
    print("\n💡 可能的解决方案:")
    print("1. 检查网络连接")
    print("2. 确认API密钥是否正确")
    print("3. 检查API是否有余额")
    print("4. 确认模型名称是否正确")
