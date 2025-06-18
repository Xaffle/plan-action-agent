"""
智能体系统 - 具有分层规划和执行能力的AI助手

本系统实现了一个能够进行高层规划(High-level Planning)和低层执行(Low-level Action)的智能体：
1. 高层规划：将复杂目标分解为具体可执行的任务序列
2. 低层执行：按顺序执行每个具体任务并收集结果
3. 状态管理：使用LangGraph管理整个工作流的状态转换

技术栈:
- LangChain: 大语言模型调用和提示管理
- LangGraph: 工作流状态图管理
- 通义千问: 底层大语言模型
"""

import os
from typing import Annotated, Sequence, TypeVar, Any, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph
import operator

# ==================== 状态定义 ====================
class AgentState(TypedDict):
    """
    智能体的状态类型定义
    
    这个状态会在整个工作流中传递和更新，包含：
    - objective: 用户的原始目标/任务描述
    - plan: 规划器生成的任务列表
    - current_step: 当前执行到第几个任务（从0开始）
    - messages: 对话历史，用于上下文传递
    - results: 每个任务的执行结果列表
    """
    objective: str          # 原始目标
    plan: list[str]        # 任务计划列表
    current_step: int      # 当前执行步骤
    messages: list[BaseMessage]  # 消息历史
    results: list[str]     # 执行结果列表

# ==================== 分层规划器 ====================
class HierarchicalPlanner:
    """
    分层规划器 - 负责高层规划(High-level Planning)
    
    功能：
    1. 接收用户的复杂目标
    2. 将目标分解为3-5个可执行的子任务
    3. 返回结构化的任务列表
    
    工作原理：
    - 使用精心设计的提示模板引导AI进行任务分解
    - 低温度(0)确保输出稳定性和一致性
    - 简单的文本解析提取任务列表
    """
    def __init__(self, llm):
        """初始化规划器"""
        self.llm = llm  # 用于规划的语言模型
        
        # 规划提示模板：引导AI进行结构化的任务分解
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a hierarchical task planner. Given a high-level objective, break it down into:
1. A sequence of high-level steps (maximum 3-5 steps)
2. For each high-level step, provide 2-3 concrete subtasks that need to be completed

Format your response as a simple numbered list of tasks:
1. [Task description]
2. [Task description]
3. [Task description]
...

Only return the numbered list, nothing else.
"""),
            MessagesPlaceholder(variable_name="chat_history"),  # 保持对话上下文
            ("human", "{input}")
        ])

    def plan(self, inputs: dict[str, Any], **kwargs) -> list[str]:
        """
        执行规划任务
        
        Args:
            inputs: 包含目标和聊天历史的字典
            
        Returns:
            list[str]: 解析后的任务列表
        """
        print("🤖 正在调用AI模型生成计划...")
        messages = self.prompt.invoke(inputs)
        result = self.llm.invoke(messages)
        print("🤖 AI模型响应完成，正在解析计划...")
        
        # 解析AI返回的文本，提取任务列表
        lines = result.content.strip().split('\n')
        tasks = []
        for line in lines:
            line = line.strip()
            # 识别编号列表格式 (1. 或 - )
            if line and (line[0].isdigit() or line.startswith('- ')):
                # 移除数字前缀和破折号，提取任务内容
                if line[0].isdigit():
                    task = line.split('.', 1)[1].strip() if '.' in line else line
                else:
                    task = line[2:].strip()
                tasks.append(task)
        print(f"📋 解析完成，提取到 {len(tasks)} 个任务")
        return tasks

# ==================== 动作执行器 ====================
class ActionExecutor:
    """
    动作执行器 - 负责低层执行(Low-level Action)
    
    功能：
    1. 接收具体的任务描述
    2. 思考执行策略并模拟执行
    3. 返回结构化的执行结果
    
    工作原理：
    - 使用"思考-行动-结果"的执行模式
    - 中等温度(0.5)保持一定的创造性
    - 结合聊天历史保持上下文连贯性
    """
    def __init__(self, llm):
        """初始化执行器"""
        self.llm = llm  # 用于执行的语言模型
        
        # 执行提示模板：引导AI进行结构化的任务执行
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an action executor. Given a specific task, you should:
1. Think about the specific actions needed
2. Execute the actions
3. Report the results and any relevant observations

Format your response as:
Thought: [Your reasoning about how to execute the task]
Action: [The specific action you're taking]
Result: [The outcome of your action]
"""),
            MessagesPlaceholder(variable_name="chat_history"),  # 保持执行上下文
            ("human", "{task}")
        ])

    def execute(self, task: str, chat_history: list[BaseMessage] = None) -> str:
        """
        执行具体任务
        
        Args:
            task: 要执行的任务描述
            chat_history: 聊天历史，用于保持上下文
            
        Returns:
            str: 执行结果的详细描述
        """
        if chat_history is None:
            chat_history = []
        print("🤖 正在调用AI模型执行任务...")
        messages = self.prompt.invoke({"task": task, "chat_history": chat_history})
        result = self.llm.invoke(messages)
        print("🤖 AI模型执行完成")
        return result.content

# ==================== 智能体工作流创建 ====================
def create_agent():
    """
    创建智能体工作流
    
    工作流程：
    1. 规划阶段(planning)：分析目标，生成任务计划
    2. 执行阶段(execute)：逐个执行任务，收集结果
    3. 决策阶段：判断是否继续执行下一个任务
    
    状态流转：
    planning -> execute -> (continue?) -> execute -> ... -> end
    
    Returns:
        StateGraph: 编译好的工作流图
    """
    # ==================== 模型初始化 ====================
    # 规划器使用低温度，确保输出稳定
    planner_llm = ChatOpenAI(
        model="qwen-plus-2025-01-25",
        temperature=0,  # 低温度确保规划的一致性
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 执行器使用中等温度，保持一定创造性
    executor_llm = ChatOpenAI(
        model="qwen-plus-2025-01-25",
        temperature=0.5,  # 中等温度保持执行的灵活性
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 创建功能组件
    planner = HierarchicalPlanner(planner_llm)
    executor = ActionExecutor(executor_llm)
    
    # ==================== 状态转换函数定义 ====================
    def plan_step(state: AgentState) -> AgentState:
        """
        规划步骤：生成任务计划
        
        这个函数只在第一次执行时生成计划，后续执行会跳过
        
        Args:
            state: 当前状态
            
        Returns:
            AgentState: 更新后的状态（包含计划）
        """
        print("🔄 开始规划阶段...")
        
        # 如果还没有计划，则生成计划
        if not state.get("plan"):
            print("📋 正在生成任务计划...")
            plan = planner.plan({
                "input": state["objective"], 
                "chat_history": state.get("messages", [])
            })
            print(f"✅ 计划生成完成，共 {len(plan)} 个任务:")
            for i, task in enumerate(plan, 1):
                print(f"   {i}. {task}")
            print("-" * 50)
            
            # 更新状态：添加计划和消息
            return {
                **state,
                "plan": plan,
                "messages": state.get("messages", []) + [HumanMessage(content=f"Created plan with {len(plan)} tasks")]
            }
        return state

    def execute_step(state: AgentState) -> AgentState:
        """
        执行步骤：执行当前任务
        
        这个函数会被重复调用，直到所有任务执行完成
        每次执行会：
        1. 检查是否还有待执行的任务
        2. 执行当前任务
        3. 更新状态（步骤计数器、结果列表、消息历史）
        
        Args:
            state: 当前状态
            
        Returns:
            AgentState: 更新后的状态
        """
        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])
        
        # 检查是否所有任务都已完成
        if current_step >= len(plan):
            print("🎉 所有任务执行完成！")
            return state
            
        # 执行当前任务
        current_task = plan[current_step]
        print(f"🚀 执行第 {current_step + 1} 个任务: {current_task}")
        print("⏳ 正在处理...")
        
        # 调用执行器执行任务
        result = executor.execute(current_task, state.get("messages", []))
        
        print(f"✅ 任务 {current_step + 1} 完成")
        print(f"📄 执行结果预览: {result[:100]}...")
        print("-" * 50)
        
        # 更新状态：增加步骤计数、添加结果、更新消息历史
        new_results = state.get("results", []) + [result]
        new_messages = state.get("messages", []) + [HumanMessage(content=f"Executed step {current_step + 1}: {result}")]
        
        return {
            **state,
            "current_step": current_step + 1,  # 递增步骤计数
            "results": new_results,            # 添加执行结果
            "messages": new_messages           # 更新消息历史
        }
    
    def should_continue(state: AgentState) -> str:
        """
        决策函数：判断工作流是否应该继续执行
        
        这是一个条件边（conditional edge）的决策函数
        根据当前执行进度决定下一步的流向
        
        Args:
            state: 当前状态
            
        Returns:
            str: "execute" 继续执行下一个任务，"end" 结束工作流
        """
        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])
        
        # 如果还有未执行的任务，继续执行
        if current_step >= len(plan):
            return "end"      # 所有任务完成，结束工作流
        return "execute"      # 继续执行下一个任务
    # ==================== 工作流图构建 ====================
    """
    工作流结构：
    
    [START] -> planning -> execute -> {decision} -> execute -> ... -> [END]
                             ^           |
                             |___________|
    
    说明：
    - planning: 只执行一次，生成任务计划
    - execute: 循环执行，每次处理一个任务
    - decision: 条件分支，决定是继续执行还是结束
    """
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 添加节点（避免与状态字段名冲突）
    workflow.add_node("planning", plan_step)    # 规划节点
    workflow.add_node("execute", execute_step)  # 执行节点
    
    # 设定边和流向
    workflow.set_entry_point("planning")        # 入口点：规划阶段
    workflow.add_edge("planning", "execute")    # 规划完成后进入执行阶段
    
    # 条件边：根据执行情况决定下一步
    workflow.add_conditional_edges(
        "execute",           # 从执行节点出发
        should_continue,     # 使用决策函数判断
        {
            "execute": "execute",  # 继续执行 -> 回到执行节点
            "end": END            # 结束 -> 工作流终止
        }
    )
    
    return workflow


# ==================== 智能体运行入口 ====================
def run_agent(objective: str):
    """
    智能体运行入口函数
    
    这是整个系统的主入口，负责：
    1. 初始化工作流
    2. 设置初始状态
    3. 执行完整的规划和执行流程
    4. 返回最终结果
    
    Args:
        objective: 用户的目标描述
        
    Returns:
        list[str]: 所有任务的执行结果列表
    """
    print("🔧 初始化智能体工作流...")
    workflow = create_agent()
    
    print("🔧 编译工作流...")
    agent = workflow.compile()  # 编译状态图为可执行的工作流
    
    print("📝 创建初始状态...")
    # 初始化状态：只设置目标，其他字段为空
    initial_state: AgentState = {
        "objective": objective,     # 用户目标
        "plan": [],                # 待生成的计划
        "current_step": 0,         # 从第0步开始
        "messages": [],            # 空的消息历史
        "results": []              # 空的结果列表
    }
    
    print("🎬 开始执行智能体...")
    print("=" * 50)
    
    # 执行工作流：传入初始状态，返回最终状态
    final_state = agent.invoke(initial_state)
    
    print("=" * 50)
    print("✨ 智能体执行完成！")
    
    # 返回所有任务的执行结果
    return final_state.get("results", [])
