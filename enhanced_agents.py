"""
增强智能体系统 - 具有自主决策和自我反思能力的AI助手（修复版）

本系统实现了一个真正由LLM驱动的智能体：
1. LLM驱动的控制流：每一步的决策都由LLM做出
2. 自我反思机制：每次执行后评估结果并调整策略
3. LangChain Agent架构：使用工具框架实现灵活的功能扩展
4. 动态计划调整：根据执行结果实时修正计划
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from api_setting import API_KEY, API_URL, API_MODEL

# ==================== 智能体状态定义 ====================
class EnhancedAgentState:
    """
    增强智能体的状态管理
    
    相比原版，增加了：
    - reflection_history: 自我反思历史
    - replan_count: 重新规划次数
    - current_focus: 当前关注点
    - confidence_score: 执行信心分数
    """
    def __init__(self, objective: str):
        self.objective = objective
        self.plan: List[str] = []
        self.completed_tasks: List[Dict] = []
        self.reflection_history: List[str] = []
        self.messages: List[BaseMessage] = []
        self.replan_count = 0
        self.current_focus = ""
        self.confidence_score = 0.0
        self.status = "initializing"  # initializing, planning, executing, reflecting, completed
    
    def to_dict(self) -> Dict:
        """将状态转换为字典，便于传递给LLM"""
        return {
            "objective": self.objective,
            "plan": self.plan,
            "completed_tasks": self.completed_tasks,
            "reflection_history": self.reflection_history,
            "replan_count": self.replan_count,
            "current_focus": self.current_focus,
            "confidence_score": self.confidence_score,
            "status": self.status,
            "progress": f"{len(self.completed_tasks)}/{len(self.plan)}" if self.plan else "0/0"
        }

# ==================== 自定义工具定义 ====================
class PlanningTool(BaseTool):
    """规划工具：生成或修正任务计划"""
    name: str = "planning_tool"
    description: str = "Create or revise a task plan based on the objective and current situation"
    
    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert task planner. Given an objective and current context, create a detailed action plan.

Your plan should:
1. Break down the objective into 3-7 concrete, actionable steps
2. Consider the current progress and any lessons learned
3. Ensure each step is specific and measurable
4. Arrange steps in logical order

Format your response as a JSON object:
{{
    "plan": [
        "Step 1: Specific action description",
        "Step 2: Another specific action"
    ],
    "reasoning": "Brief explanation of your planning approach",
    "estimated_difficulty": "low/medium/high"
}}
"""),
            ("human", "Objective: {objective}\nCurrent Context: {context}")
        ])
    
    def _run(self, objective: str, context: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        messages = self._prompt.invoke({"objective": objective, "context": context})
        result = self._llm.invoke(messages)
        # 清理可能存在的 Markdown 代码块
        raw_content = result.content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_content)
        cleaned = re.sub(r"```$", "", cleaned).strip()
        return cleaned

class ExecutionTool(BaseTool):
    """执行工具：执行具体任务"""
    name: str = "execution_tool"
    description: str = "Execute a specific task and return detailed results"
    
    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert task executor. Given a specific task, execute it thoroughly and provide detailed results.

Your execution should include:
1. Clear understanding of what needs to be done
2. Step-by-step execution process
3. Specific outcomes and observations
4. Any challenges encountered
5. Quality assessment of the results

Format your response as a JSON object:
{{
    "execution_process": "Detailed description of how you executed the task",
    "results": "Specific outcomes and deliverables",
    "challenges": "Any difficulties or obstacles encountered",
    "quality_score": 0.8,
    "recommendations": "Suggestions for improvement or next steps"
}}
"""),
            ("human", "Task: {task}\nContext: {context}")
        ])
    
    def _run(self, task: str, context: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        messages = self._prompt.invoke({"task": task, "context": context})
        result = self._llm.invoke(messages)
        # 清理可能存在的 Markdown 代码块
        raw_content = result.content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_content)
        cleaned = re.sub(r"```$", "", cleaned).strip()
        return cleaned

class ReflectionTool(BaseTool):
    """反思工具：评估执行结果并提供改进建议"""
    name: str = "reflection_tool"
    description: str = "Reflect on recent execution results and provide insights for improvement"
    
    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert performance evaluator. Analyze recent task execution and provide valuable insights.

Your reflection should include:
1. Assessment of what went well and what didn't
2. Identification of patterns or recurring issues
3. Specific recommendations for improvement
4. Strategic insights for future planning
5. Confidence level in current approach

Format your response as a JSON object:
{{
    "assessment": "Overall evaluation of recent performance",
    "strengths": ["What worked well"],
    "weaknesses": ["Areas for improvement"],
    "patterns": ["Notable patterns observed"],
    "recommendations": ["Specific suggestions for improvement"],
    "confidence_adjustment": 0.1,
    "should_replan": true
}}
"""),
            ("human", "Recent Execution History: {history}\nCurrent Plan: {plan}\nObjective: {objective}")
        ])
    
    def _run(self, history: str, plan: str, objective: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        messages = self._prompt.invoke({"history": history, "plan": plan, "objective": objective})
        result = self._llm.invoke(messages)
        # 清理可能存在的 Markdown 代码块
        raw_content = result.content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_content)
        cleaned = re.sub(r"```$", "", cleaned).strip()
        return cleaned

# ==================== LLM驱动的控制器 ====================
class LLMController:
    """
    LLM驱动的智能体控制器
    
    核心特性：
    1. 每一步的决策都由LLM做出
    2. 根据当前状态动态选择下一个行动
    3. 支持循环、跳跃、重新规划等复杂控制流
    """
    def __init__(self, llm):
        self.llm = llm
        self.decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the central controller of an intelligent agent system. Your role is to analyze the current situation and decide the next best action.

Available actions:
1. "plan" - Create or revise the task plan
2. "execute" - Execute the next task in the plan
3. "reflect" - Analyze recent performance and adjust strategy
4. "replan" - Completely revise the plan based on new insights
5. "complete" - Mark the objective as completed

Consider:
- Current progress and remaining work
- Quality of recent executions
- Whether the current plan is still optimal
- Any patterns in the execution history

Respond with a JSON object:
{{
    "action": "plan|execute|reflect|replan|complete",
    "reasoning": "Detailed explanation of why this action is best",
    "parameters": {{}},
    "confidence": 0.8,
    "urgency": "low|medium|high"
}}"""),
            ("human", "Current State: {state}\nRecent History: {history}")
        ])
    
    def decide_next_action(self, state: EnhancedAgentState) -> Dict:
        """让LLM决定下一步行动"""
        # 准备状态信息
        state_info = state.to_dict()
        
        # 准备最近的历史信息
        recent_history = []
        if state.completed_tasks:
            recent_history = state.completed_tasks[-3:]  # 最近3个任务
        if state.reflection_history:
            recent_history.extend(state.reflection_history[-2:])  # 最近2次反思
        
        # 调用LLM进行决策
        messages = self.decision_prompt.invoke({
            "state": json.dumps(state_info, indent=2, ensure_ascii=False),
            "history": json.dumps(recent_history, indent=2, ensure_ascii=False)
        })
        
        result = self.llm.invoke(messages)
        
        # 清理可能存在的 Markdown 代码块
        raw_content = result.content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_content)
        cleaned = re.sub(r"```$", "", cleaned).strip()
        
        try:
            decision = json.loads(cleaned)
            return decision
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回默认决策
            return {
                "action": "reflect",
                "reasoning": "Failed to parse decision, defaulting to reflection",
                "parameters": {},
                "confidence": 0.3,
                "urgency": "medium"
            }

# ==================== 增强智能体主类 ====================
class EnhancedAgent:
    """
    增强版智能体 - 具有自主决策和自我反思能力
    
    核心改进：
    1. LLM驱动的控制流
    2. 自我反思机制
    3. 基于LangChain工具的模块化架构
    """
    def __init__(self):
        # 初始化LLM（稍高的温度以支持创造性决策）
        self.llm = ChatOpenAI(
            model=API_MODEL,
            temperature=0.3,
            api_key=API_KEY,
            base_url=API_URL
        )
        
        # 初始化控制器
        self.controller = LLMController(self.llm)
        
        # 初始化工具
        self.planning_tool = PlanningTool(self.llm)
        self.execution_tool = ExecutionTool(self.llm)
        self.reflection_tool = ReflectionTool(self.llm)
        
        # 创建工具列表
        self.tools = [
            self.planning_tool,
            self.execution_tool,
            self.reflection_tool
        ]
    
    def run(self, objective: str, max_iterations: int = 20) -> Dict:
        """
        运行增强智能体
        
        Args:
            objective: 用户目标
            max_iterations: 最大迭代次数
            
        Returns:
            Dict: 包含执行结果的字典
        """
        print("🚀 启动增强智能体...")
        print(f"🎯 目标: {objective}")
        print("=" * 60)
        
        # 初始化状态
        state = EnhancedAgentState(objective)
        iteration_count = 0
        
        while iteration_count < max_iterations:
            iteration_count += 1
            print(f"\n🔄 第 {iteration_count} 轮决策")
            
            # LLM决策下一步行动
            decision = self.controller.decide_next_action(state)
            action = decision.get("action", "reflect")
            reasoning = decision.get("reasoning", "No reasoning provided")
            confidence = decision.get("confidence", 0.5)
            
            print(f"🤔 LLM决策: {action}")
            print(f"💭 决策理由: {reasoning}")
            print(f"🎯 信心分数: {confidence:.2f}")
            
            # 更新状态
            state.confidence_score = confidence
            state.status = action
            
            # 执行相应动作
            if action == "plan" or action == "replan":
                self._handle_planning(state, action == "replan")
                
            elif action == "execute":
                self._handle_execution(state)
                
            elif action == "reflect":
                self._handle_reflection(state)
                
            elif action == "complete":
                print("✅ LLM判断目标已完成！")
                break
                
            else:
                print(f"⚠️ 未知动作: {action}，切换到反思模式")
                self._handle_reflection(state)
            
            # 安全检查：如果陷入循环，强制退出
            if iteration_count >= max_iterations:
                print("⚠️ 达到最大迭代次数，强制结束")
                break
        
        print("\n" + "=" * 60)
        print("🎉 智能体执行完成！")
        
        # 返回最终结果
        return {
            "objective": objective,
            "completed_tasks": state.completed_tasks,
            "reflections": state.reflection_history,
            "final_confidence": state.confidence_score,
            "iterations_used": iteration_count,
            "status": "completed" if action == "complete" else "max_iterations_reached"
        }
    
    def _handle_planning(self, state: EnhancedAgentState, is_replan: bool = False):
        """处理规划动作"""
        action_type = "重新规划" if is_replan else "初始规划"
        print(f"📋 执行{action_type}...")
        
        # 准备上下文
        context = {
            "completed_tasks": state.completed_tasks,
            "reflections": state.reflection_history,
            "replan_count": state.replan_count
        }
        
        # 调用规划工具
        result = self.planning_tool._run(
            objective=state.objective,
            context=json.dumps(context, ensure_ascii=False)
        )
        
        try:
            plan_data = json.loads(result)
            state.plan = plan_data.get("plan", [])
            
            if is_replan:
                state.replan_count += 1
            
            print(f"✅ {action_type}完成，共 {len(state.plan)} 个任务:")
            for i, task in enumerate(state.plan, 1):
                print(f"   {i}. {task}")
            
            # 记录到消息历史
            state.messages.append(HumanMessage(content=f"{action_type}: {len(state.plan)} tasks created"))
            
        except json.JSONDecodeError:
            print("❌ 规划结果解析失败，保持原计划")
    
    def _handle_execution(self, state: EnhancedAgentState):
        """处理执行动作"""
        if not state.plan:
            print("❌ 没有可执行的计划，切换到规划模式")
            self._handle_planning(state)
            return
        
        # 找到下一个未完成的任务
        next_task_index = len(state.completed_tasks)
        if next_task_index >= len(state.plan):
            print("✅ 所有计划任务已完成")
            return
        
        current_task = state.plan[next_task_index]
        print(f"🚀 执行任务 {next_task_index + 1}: {current_task}")
        
        # 准备执行上下文
        context = {
            "completed_tasks": state.completed_tasks,
            "remaining_tasks": state.plan[next_task_index + 1:],
            "objective": state.objective
        }
        
        # 调用执行工具
        result = self.execution_tool._run(
            task=current_task,
            context=json.dumps(context, ensure_ascii=False)
        )
        
        try:
            execution_data = json.loads(result)
            
            # 记录执行结果
            task_result = {
                "task": current_task,
                "execution_process": execution_data.get("execution_process", ""),
                "results": execution_data.get("results", ""),
                "challenges": execution_data.get("challenges", ""),
                "quality_score": execution_data.get("quality_score", 0.5),
                "recommendations": execution_data.get("recommendations", "")
            }
            
            state.completed_tasks.append(task_result)
            
            print(f"✅ 任务完成，质量分数: {task_result['quality_score']:.2f}")
            # 确保 results 是字符串再进行切片
            results_str = str(task_result.get('results', ''))
            print(f"📋 结果摘要: {results_str[:100]}...")
            
            # 记录到消息历史
            state.messages.append(HumanMessage(content=f"Completed task: {current_task}"))
            
        except json.JSONDecodeError:
            print("❌ 执行结果解析失败，记录原始结果")
            state.completed_tasks.append({
                "task": current_task,
                "results": result,
                "quality_score": 0.3
            })
    
    def _handle_reflection(self, state: EnhancedAgentState):
        """处理反思动作"""
        print("🤔 执行自我反思...")
        
        if not state.completed_tasks:
            print("⚠️ 暂无执行历史，跳过反思")
            return
        
        # 调用反思工具
        result = self.reflection_tool._run(
            history=json.dumps(state.completed_tasks, ensure_ascii=False),
            plan=json.dumps(state.plan, ensure_ascii=False),
            objective=state.objective
        )
        
        try:
            reflection_data = json.loads(result)
            
            # 记录反思结果
            reflection_summary = {
                "assessment": reflection_data.get("assessment", ""),
                "strengths": reflection_data.get("strengths", []),
                "weaknesses": reflection_data.get("weaknesses", []),
                "recommendations": reflection_data.get("recommendations", []),
                "should_replan": reflection_data.get("should_replan", False)
            }
            
            state.reflection_history.append(json.dumps(reflection_summary, ensure_ascii=False))
              # 调整信心分数
            confidence_adjustment = reflection_data.get("confidence_adjustment", 0)
            state.confidence_score = max(0, min(1, state.confidence_score + confidence_adjustment))
            
            print(f"✅ 反思完成")
            # 确保 assessment 是字符串再进行切片
            assessment_str = str(reflection_summary.get('assessment', ''))
            print(f"📈 评估: {assessment_str[:100]}...")
            if reflection_summary['strengths']:
                print(f"💪 优势: {', '.join(reflection_summary['strengths'][:2])}")
            if reflection_summary['weaknesses']:
                print(f"🔧 改进点: {', '.join(reflection_summary['weaknesses'][:2])}")
            
            if reflection_summary['should_replan']:
                print("🔄 反思建议重新规划")
            
            # 记录到消息历史
            state.messages.append(HumanMessage(content="Completed self-reflection"))
            
        except json.JSONDecodeError:
            print("❌ 反思结果解析失败")

# ==================== 运行入口函数 ====================
def run_enhanced_agent(objective: str, max_iterations: int = 20) -> Dict:
    """
    运行增强智能体的入口函数
    
    Args:
        objective: 用户目标
        max_iterations: 最大迭代次数
        
    Returns:
        Dict: 执行结果
    """
    agent = EnhancedAgent()
    return agent.run(objective, max_iterations)

# ==================== 测试函数 ====================
if __name__ == "__main__":
    # 简单测试
    test_objective = "制定一个周末学习Python数据分析的计划，并开始第一步学习"
    result = run_enhanced_agent(test_objective, max_iterations=10)
    
    print("\n" + "=" * 60)
    print("📊 最终结果摘要:")
    print(f"🎯 目标: {result['objective']}")
    print(f"✅ 完成任务数: {len(result['completed_tasks'])}")
    print(f"🤔 反思次数: {len(result['reflections'])}")
    print(f"💯 最终信心: {result['final_confidence']:.2f}")
    print(f"🔄 使用迭代: {result['iterations_used']}")
    print(f"📈 状态: {result['status']}")
