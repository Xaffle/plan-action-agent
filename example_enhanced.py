"""
增强智能体使用示例

展示如何使用新的增强智能体系统
"""

from enhanced_agents import run_enhanced_agent

def main():
    print("🤖 增强智能体演示")
    print("=" * 50)
    
    # 示例1: 学习规划
    print("\n📚 示例1: 学习规划助手")
    learning_objective = "帮我制定一个掌握机器学习基础的3天学习计划，并开始第一天的学习"
    
    result = run_enhanced_agent(learning_objective, max_iterations=10)
    
    # 示例2: 项目规划
    print("\n" + "="*50)
    print("\n🚀 示例2: 项目规划助手")
    project_objective = "为一个在线书店系统设计完整的开发计划，包括技术架构、数据库设计和开发流程"
    
    result2 = run_enhanced_agent(project_objective, max_iterations=12)
    
    print("\n🎯 演示完成！")
    print(f"第一个任务完成了 {len(result['completed_tasks'])} 个子任务")
    print(f"第二个任务完成了 {len(result2['completed_tasks'])} 个子任务")

if __name__ == "__main__":
    main()
