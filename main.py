import os
from agents import run_agent

# 确保已设置 DASHSCOPE_API_KEY 环境变量
if not os.environ.get("DASHSCOPE_API_KEY"):
    raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

# 测试智能体
objective = """请帮我规划并执行一个新产品发布的社交媒体营销计划。要求：
1. 目标人群是25-35岁的年轻专业人士
2. 产品是一款智能健康手表
3. 需要在一周内完成从内容策划到发布的全过程
"""

results = run_agent(objective)

# 打印执行结果
print("\n=== 执行结果 ===")
for i, result in enumerate(results, 1):
    print(f"\nStep {i}:")
    print(result)
