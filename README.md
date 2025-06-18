# Demo Agent with plan and action

## 🎯 系统概述

这是一个具有**分层规划**和**低层执行**能力的智能体系统，能够将复杂目标分解为可执行的任务序列，并逐步完成。系统基于 LangGraph 构建，实现了状态管理和工作流控制。

## 🏗️ 系统架构

```
用户输入目标
    ↓
┌──────────────────┐
│   智能体入口     │ ← run_agent()
│  (初始化&编排)   │
└──────────────────┘
    ↓
┌──────────────────┐
│   规划阶段       │ ← HierarchicalPlanner
│  (目标→任务列表) │    - 高层规划 (温度=0)
└──────────────────┘    - 结构化任务分解
    ↓
┌──────────────────┐
│   执行阶段       │ ← ActionExecutor
│  (逐个执行任务)  │    - 具体执行 (温度=0.5)
└──────────────────┘    - 思考-行动-结果模式
    ↓
┌──────────────────┐
│   决策阶段       │ ← should_continue()
│  (继续或结束?)   │    - 循环控制逻辑
└──────────────────┘
    ↓
   结果输出
```

## 🔧 核心组件

### 1. 状态管理 (AgentState)
```python
class AgentState(TypedDict):
    objective: str              # 用户原始目标
    plan: list[str]            # 任务计划列表
    current_step: int          # 当前执行步骤
    messages: list[BaseMessage] # 对话历史
    results: list[str]         # 执行结果
```

### 2. 分层规划器 (HierarchicalPlanner)
- **职责**: 高层规划 (High-level Planning)
- **输入**: 用户目标描述
- **输出**: 结构化任务列表 (3-5个任务)
- **特点**: 
  - 低温度(0)确保输出稳定
  - 使用编号列表格式解析
  - 支持聊天历史上下文

### 3. 动作执行器 (ActionExecutor)
- **职责**: 低层执行 (Low-level Action)  
- **输入**: 具体任务描述
- **输出**: 思考-行动-结果格式的执行报告
- **特点**: 
  - 中等温度(0.5)保持创造性
  - 结构化输出格式
  - 上下文连贯性

### 4. 工作流引擎 (LangGraph StateGraph)
- **职责**: 状态管理和流程控制
- **特点**: 
  - 支持循环执行和条件分支
  - 状态在节点间传递和更新
  - 编译为可执行的工作流图
- **流程**: planning → execute → decision → execute...

## 🔄 执行流程

### 工作流节点说明
1. **planning节点**: 
   - 只执行一次
   - 调用 `HierarchicalPlanner` 生成任务列表
   - 更新状态中的 `plan` 字段

2. **execute节点**: 
   - 循环执行，每次处理一个任务
   - 调用 `ActionExecutor` 执行当前任务
   - 更新 `current_step`、`results`、`messages`

3. **条件边逻辑**:
   - `should_continue()` 判断是否继续
   - 返回 "execute" 继续下一任务
   - 返回 "end" 结束工作流

### 详细执行步骤
1. **初始化**: 设置目标，创建空白状态
2. **规划阶段**: AI分析目标，生成3-5个任务
3. **执行循环**: 
   - 执行当前任务 (步骤 current_step)
   - 收集执行结果到 results 列表
   - 更新消息历史
   - 递增步骤计数器
   - 判断是否继续
4. **结果汇总**: 返回所有任务的执行结果

## 🎯 关键特性

### 分层处理
- **高层**: 战略规划，目标分解
- **低层**: 具体执行，结果反馈

### 状态持久化  
- 完整的执行历史
- 上下文连贯性
- 错误恢复能力

### 灵活扩展
- 可替换不同的LLM模型
- 可自定义规划和执行策略
- 支持复杂的条件分支

## 🚀 使用方法

### 1. 环境设置
```bash
# 设置通义千问API密钥
export DASHSCOPE_API_KEY="your-dashscope-api-key"

# 或者设置API提供商
export API_PROVIDER="qwen"  # 默认值，可选 "qwen" 或 "deepseek"
```

### 2. API配置
系统支持多个API提供商，通过 `api_setting.py` 统一管理：

```python
# api_setting.py 中的配置
API_PROVIDER = "qwen"  # 或 "deepseek"

if API_PROVIDER == "qwen":
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    API_MODEL = "qwen3-30b-a3b"
elif API_PROVIDER == "deepseek":
    API_KEY = os.getenv("ALT_API_KEY")
    API_URL = "https://alt-llm.example.com/api/v1"
    API_MODEL = "alt-model-1"
```

### 3. 运行智能体
```python
# 基本用法
from agents import run_agent

objective = """请帮我规划并执行一个新产品发布的社交媒体营销计划。要求：
1. 目标人群是25-35岁的年轻专业人士
2. 产品是一款智能健康手表
3. 需要在一周内完成从内容策划到发布的全过程
"""

results = run_agent(objective)

# 查看结果
for i, result in enumerate(results, 1):
    print(f"任务{i}: {result}")
```

### 4. 测试脚本
```bash
# 运行主程序示例
python main.py

# 运行简单测试
python test_simple.py

# 测试LLM连接
python test_llm.py
```

## 💡 适用场景

- **项目规划和执行**: 复杂项目的分解和实施
- **学习计划制定**: 知识体系的系统化学习安排
- **工作流程自动化**: 业务流程的智能化处理
- **营销策划**: 多渠道营销活动的规划执行
- **产品发布**: 从策划到上线的全流程管理
- **多步骤问题解决**: 复杂问题的结构化分析

## ⚡ 系统特性

### 分层处理架构
- **高层规划**: 战略分解，全局视角 (温度=0，确保稳定)
- **低层执行**: 具体实施，灵活应变 (温度=0.5，保持创造性)

### 状态持久化管理
- 完整的执行历史追踪
- 上下文连贯性保证
- 任务间信息传递
- 错误恢复能力

### 灵活的扩展性
- 可替换不同的LLM模型和提供商
- 可自定义规划和执行策略  
- 支持复杂的条件分支逻辑
- 模块化设计，易于维护

### 工作流可视化
- 清晰的状态转换逻辑
- 循环执行控制
- 条件分支决策
- 实时执行进度反馈

## 🚨 注意事项

1. **API密钥配置**: 确保正确设置环境变量
2. **网络连接**: 需要稳定的网络访问API服务
3. **任务复杂度**: 建议单个目标包含3-7个子任务
4. **执行时间**: 复杂任务可能需要较长执行时间
5. **成本控制**: 注意API调用频次和成本

## 📝 更新日志

- **v1.0**: 基础分层规划和执行功能
- 支持通义千问API集成
- 实现LangGraph状态管理
- 添加多API提供商支持

## 🔧 技术栈

- **LangChain**: LLM调用和提示管理
- **LangGraph**: 工作流状态图管理
- **通义千问 (Qwen)**: 底层大语言模型
  - 规划器: `qwen-plus-2025-01-25` (温度=0)
  - 执行器: `qwen-plus-2025-01-25` (温度=0.5)
- **Python**: 系统实现语言
- **TypedDict**: 状态类型定义

## 📁 项目结构

```
plan-action-agent/
├── agents.py          # 核心智能体实现
├── api_setting.py     # API配置管理
├── main.py           # 主程序示例
├── test_llm.py       # LLM连接测试
├── test_simple.py    # 简单功能测试
└── README.md         # 项目说明文档
```

## 🔍 核心文件说明

### agents.py
- `AgentState`: 状态类型定义
- `HierarchicalPlanner`: 分层规划器类
- `ActionExecutor`: 动作执行器类
- `create_agent()`: 工作流创建函数
- `run_agent()`: 主入口函数

### api_setting.py  
- 统一的API配置管理
- 支持多个LLM提供商切换
- 环境变量检查和错误处理

### 测试文件
- `main.py`: 完整的使用示例
- `test_simple.py`: 基础功能测试
- `test_llm.py`: API连接验证
