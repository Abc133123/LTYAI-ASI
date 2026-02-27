#!/usr/bin/env python3
"""
天依AI 3.2 - 预测思维版AGI（修复版）
核心理念：世界是自然语言文本，LLM通过文本推演世界，学习就是修改文本,LLM是老师，不是大脑。智能体通过内在驱动在世界中探索，从意外中学习规则

修复内容：
1. 修复饿死导致程序退出 -> 改为迭代重置
2. 程序无限循环运行
3. 将"天"改成"轮"
4. 将"饿死"改成"迭代"
5. 添加知识库持久化（JSON）
6. 自动补充食物机制
7. 修复代码格式错误，添加科幻风格彩色打印
项目需要deepseekAPI 脚本内搜索''替换文本''
"""

import json
import time
import random
import re
import os
from openai import OpenAI
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import traceback

# 导入彩色打印库（科幻风格配色）
from colorama import init, Fore, Back, Style

# 初始化colorama（兼容Windows）
init(autoreset=True)

# ================= 科幻风格彩色打印工具 =================
class SciFiPrinter:
    """科幻风格彩色打印工具类"""
    # 科幻配色定义
    SYSTEM_COLOR = Fore.CYAN + Style.BRIGHT    # 系统信息 - 亮青色
    ENV_COLOR = Fore.LIGHTGREEN_EX + Style.BRIGHT  # 环境信息 - 亮绿色
    DECISION_COLOR = Fore.MAGENTA + Style.BRIGHT   # 决策信息 - 亮洋红
    ACTION_COLOR = Fore.YELLOW + Style.BRIGHT      # 行动信息 - 亮黄色
    LEARN_COLOR = Fore.BLUE + Style.BRIGHT         # 学习信息 - 亮蓝色
    SURPRISE_COLOR = Fore.RED + Style.BRIGHT       # 意外信息 - 亮红色
    SUMMARY_COLOR = Fore.WHITE + Back.BLUE + Style.BRIGHT  # 总结信息 - 白字蓝底
    THOUGHT_COLOR = Fore.LIGHTMAGENTA_EX           # 内心独白 - 浅洋红
    KNOWLEDGE_COLOR = Fore.LIGHTCYAN_EX + Style.BRIGHT  # 知识库 - 浅青

    @staticmethod
    def print_system(msg: str):
        """打印系统信息"""
        print(f"{SciFiPrinter.SYSTEM_COLOR}[系统] {msg}")

    @staticmethod
    def print_env(msg: str):
        """打印环境信息"""
        print(f"{SciFiPrinter.ENV_COLOR}[环境] {msg}")

    @staticmethod
    def print_decision(msg: str):
        """打印决策信息"""
        print(f"{SciFiPrinter.DECISION_COLOR}[决策] {msg}")

    @staticmethod
    def print_action(msg: str):
        """打印行动信息"""
        print(f"{SciFiPrinter.ACTION_COLOR}[行动] {msg}")

    @staticmethod
    def print_learn(msg: str):
        """打印学习信息"""
        print(f"{SciFiPrinter.LEARN_COLOR}[学习] {msg}")

    @staticmethod
    def print_surprise(msg: str):
        """打印意外信息"""
        print(f"{SciFiPrinter.SURPRISE_COLOR}[意外] {msg}")

    @staticmethod
    def print_summary(msg: str):
        """打印总结信息"""
        print(f"{SciFiPrinter.SUMMARY_COLOR}[总结] {msg}")

    @staticmethod
    def print_thought(msg: str):
        """打印内心独白"""
        print(f"{SciFiPrinter.THOUGHT_COLOR}   内心独白：{msg}")

    @staticmethod
    def print_knowledge(msg: str):
        """打印知识库信息"""
        print(f"{SciFiPrinter.KNOWLEDGE_COLOR}[知识库] {msg}")

# ================= 配置 =================
DEEPSEEK_API_KEY = "替换文本"  # 请替换为实际的deepseekAPI Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
KNOWLEDGE_FILE = "agi_knowledge.json"
MODEL_CACHE_DIR = "./model_cache"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================= 符号规则引擎 v1.0 =================
@dataclass
class SymbolicRule:
    id: str
    conditions: Dict[str, Any]
    effects: Dict[str, Any]
    confidence: float = 1.0
    source: str = "手动定义"

class SymbolicWorldModel:
    """符号规则引擎，保证决策的可解释性和可靠性"""
    
    def __init__(self):
        self.rules: List[SymbolicRule] = []
        self._load_default_rules()
        self._load_knowledge()
        
    def _load_default_rules(self):
        """加载基础生存规则"""
        default_rules = [
            SymbolicRule(
                id="R_生存_吃_可食用",
                conditions={"动作": "吃", "物体属性": "可食用"},
                effects={"状态变化": "消失", "奖励": 0.5, "描述": "你吃掉了{物体}，感觉好多了。"}
            ),
            SymbolicRule(
                id="R_探索_观察",
                conditions={"动作": "观察"},
                effects={"奖励": 0.1, "描述": "你仔细观察了{物体}。"}
            ),
            SymbolicRule(
                id="R_危险_触摸_热",
                conditions={"动作": "触摸", "物体属性": "热"},
                effects={"状态变化": "受伤", "奖励": -0.5, "描述": "你被{物体}烫伤了！"}
            ),
            SymbolicRule(
                id="R_探索_新区域",
                conditions={"动作": "探索", "目标": "未知区域"},
                effects={"奖励": 0.3, "描述": "你发现了新区域：{区域名}"}
            )
        ]
        for rule in default_rules:
            self.rules.append(rule)
    
    def _load_knowledge(self):
        """从JSON加载已学习的知识"""
        if os.path.exists(KNOWLEDGE_FILE):
            try:
                with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for rule_data in data.get('rules', []):
                        rule = SymbolicRule(
                            id=rule_data['id'],
                            conditions=rule_data['conditions'],
                            effects=rule_data['effects'],
                            confidence=rule_data.get('confidence', 0.8),
                            source=rule_data.get('source', 'LLM学习')
                        )
                        # 避免重复加载
                        if rule.id not in [r.id for r in self.rules]:
                            self.rules.append(rule)
                SciFiPrinter.print_knowledge(f"已加载 {len([r for r in self.rules if r.source == 'LLM学习'])} 条学习规则")
            except Exception as e:
                SciFiPrinter.print_knowledge(f"加载失败: {e}")
    
    def save_knowledge(self):
        """保存知识到JSON"""
        try:
            # 只保存LLM学习的规则
            learned_rules = [
                {
                    'id': rule.id,
                    'conditions': rule.conditions,
                    'effects': rule.effects,
                    'confidence': rule.confidence,
                    'source': rule.source
                }
                for rule in self.rules if rule.source == "LLM学习"
            ]
            
            data = {
                'rules': learned_rules,
                'last_update': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(KNOWLEDGE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            SciFiPrinter.print_knowledge(f"已保存 {len(learned_rules)} 条规则到 {KNOWLEDGE_FILE}")
        except Exception as e:
            SciFiPrinter.print_knowledge(f"保存失败: {e}")
    
    def predict(self, world_text: str, action: str, target: str, nearby_objects: Dict) -> Dict:
        """基于符号规则的预测"""
        # 提取目标物体属性
        target_attrs = []
        if target in nearby_objects:
            obj_data = nearby_objects[target]
            # 兼容两种数据格式
            if isinstance(obj_data, dict):
                target_attrs = obj_data.get("属性", [])
            elif isinstance(obj_data, list):
                target_attrs = obj_data
        
        # 匹配规则
        matched_rule = None
        for rule in self.rules:
            cond = rule.conditions
            
            # 检查动作
            if cond.get("动作") != action:
                continue
                
            # 检查目标类型
            if cond.get("目标") == "未知区域" and target != "未知区域":
                continue
                
            # 检查物体属性
            required_attr = cond.get("物体属性")
            if required_attr and required_attr not in target_attrs:
                continue
                
            matched_rule = rule
            break
        
        if matched_rule:
            effects = matched_rule.effects
            desc = effects.get("描述", "").replace("{物体}", target).replace("{区域名}", "新区域")
            
            return {
                "success": True,
                "reward": effects.get("奖励", 0.0),
                "desc": desc,
                "rule_id": matched_rule.id,
                "effects": effects
            }
        
        return {"success": False, "reason": "没有匹配的规则"}

# ================= 文本世界模型 =================
class TextWorldModel:
    """自然语言世界模型，LLM在此上做推演"""
    
    def __init__(self):
        self.world_narrative = """
        你站在一片空旷的起点，四周弥漫着淡淡的雾气。
        地面上散落着几个物体：一个红彤彤的苹果、一块灰色的石头、一汪清澈的水。
        空气中带着一丝甜味，天空是蔚蓝的，远方似乎有山的轮廓。
        """
        
    def update_world_text(self, action: str, target: str, result: str):
        """更新世界文本描述"""
        change_descriptions = {
            "吃": f"你吃掉了{target}，它消失了。空气中还残留着一丝香气。",
            "探索": f"你踏上了新的土地，周围的环境发生了变化。{result}",
            "观察": f"你仔细观察了{target}，发现了更多细节。"
        }
        
        change_text = change_descriptions.get(action, f"你对{target}做了{action}。")
        self.world_narrative += f"\n\n{change_text}"
        
        # 保持叙述简洁
        paragraphs = self.world_narrative.split('\n\n')
        if len(paragraphs) > 10:
            self.world_narrative = '\n\n'.join(paragraphs[-10:])
    
    def get_world_state_text(self) -> str:
        """获取当前世界状态的文本描述"""
        return self.world_narrative
    
    def reset(self):
        """重置世界描述"""
        self.world_narrative = """
        你站在一片空旷的起点，四周弥漫着淡淡的雾气。
        地面上散落着几个物体：一个红彤彤的苹果、一块灰色的石头、一汪清澈的水。
        空气中带着一丝甜味，天空是蔚蓝的，远方似乎有山的轮廓。
        """

# ================= 内心独白系统 =================
class InnerVoiceSystem:
    """生成智能体的内心独白，赋予生命力"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.personality = "高中生洛天依，温柔、好奇、略带文艺气息"
    
    def generate_thought(self, world_text: str, internal_state: Dict, nearby_objects: Dict, decision: Tuple[str, str]) -> str:
        """生成内心独白"""
        target, action = decision
        
        prompt = f"""
你是{self.personality}。现在，你需要用第一人称简短地表达你的内心想法。

当前世界：
{world_text}

周围物体：{list(nearby_objects.keys())}
内在状态：
- 饥饿：{internal_state['饥饿']:.1f}
- 好奇：{internal_state['好奇']:.1f}
- 恐惧：{internal_state['恐惧']:.1f}

你决定对 '{target}' 执行 '{action}' 动作。

请用2-3句话表达你现在的内心想法，要包含：
1. 对当前环境的感受
2. 决策的理由
3. 对未来的期待或担忧

语气要温柔、简短，像是在自言自语。直接输出独白内容，不要有其他格式。
"""
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个温柔的高中女生，正在自言自语。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                max_tokens=150
            )
            
            thought = response.choices[0].message.content.strip()
            return thought
            
        except Exception as e:
            # 回退到基础独白
            return self._basic_thought(internal_state, action, target)
    
    def _basic_thought(self, internal_state: Dict, action: str, target: str) -> str:
        """基础内心独白生成"""
        thoughts = []
        
        if internal_state['饥饿'] > 0.7:
            thoughts.append(f"肚子好饿...必须找到食物。")
        elif internal_state['好奇'] > 0.8:
            thoughts.append(f"好想知道{target}藏着什么秘密...")
        
        if action == "吃":
            thoughts.append(f"这个{target}看起来很好吃呢。")
        elif action == "探索":
            thoughts.append(f"前方会有什么呢？有点期待又有点紧张。")
        
        return " ".join(thoughts) if thoughts else f"我决定对{target}做{action}。"

# ================= 环境层 =================
class BaiduTextWorld:
    def __init__(self):
        self.visited_topics = set()
        self.current_location = "起始点"
        self.nearby_objects = {}  # 统一使用列表格式存储属性
        self.discovered_areas = {}
        self.iteration_count = 0  # 迭代计数
        
    def explore_topic(self, topic: str) -> Dict:
        """探索新区域"""
        if topic in self.visited_topics:
            return {"objects": [], "desc": "这个区域已经探索过了。"}
        
        SciFiPrinter.print_env(f"正在探索新区域：{topic}")
        
        # 模拟百科探索
        try:
            # 简化版探索，实际应用中可以接入真实API
            time.sleep(0.3)  # 模拟网络延迟
            
            # 生成区域描述
            area_descriptions = [
                f"你来到了一片{topic}区域，空气中弥漫着不同的气息。",
                f"这里似乎有很多与{topic}相关的事物。",
                f"远处传来阵阵声响，像是{topic}在召唤你。"
            ]
            
            area_desc = random.choice(area_descriptions)
            self.discovered_areas[topic] = area_desc
            self.visited_topics.add(topic)
            
            # 生成新物体
            objects = self._generate_objects_for_topic(topic)
            for obj in objects:
                obj_name = obj["名称"]
                if obj_name not in self.nearby_objects:
                    # 统一使用列表格式存储属性
                    self.nearby_objects[obj_name] = obj["属性"]
                    print(f"   {SciFiPrinter.ENV_COLOR}发现新物体：{obj_name}")
            
            self.current_location = topic
            return {
                "objects": objects,
                "desc": area_desc
            }
            
        except Exception as e:
            print(f"   {SciFiPrinter.ENV_COLOR}探索失败: {e}")
            return {"objects": [], "desc": "一片虚无，什么都没有。"}
    
    def _generate_objects_for_topic(self, topic: str) -> List[Dict]:
        """根据主题生成相关物体"""
        topic_object_map = {
            "量子": [
                {"名称": "量子碎片", "属性": ["发光", "神秘", "量子态"]},
                {"名称": "能量晶体", "属性": ["能量", "透明", "量子"]}
            ],
            "自然": [
                {"名称": "花朵", "属性": ["美丽", "可观赏", "植物"]},
                {"名称": "树叶", "属性": ["绿色", "自然", "植物"]}
            ],
            "工具": [
                {"名称": "螺丝刀", "属性": ["工具", "金属", "坚硬"]},
                {"名称": "扳手", "属性": ["工具", "金属", "坚固"]}
            ],
            "食物": [
                {"名称": "面包", "属性": ["可食用", "食物", "固体"]},
                {"名称": "水果", "属性": ["可食用", "水果", "甜美"]}
            ]
        }
        
        # 根据关键词匹配
        for key, objects in topic_object_map.items():
            if key in topic:
                return objects
        
        # 默认物体
        return [
            {"名称": f"{topic}样本", "属性": ["未知", "新奇"]},
            {"名称": "神秘石头", "属性": ["坚硬", "神秘"]}
        ]
    
    def get_nearby_objects(self) -> Dict:
        return self.nearby_objects
    
    def remove_object(self, obj_name: str):
        if obj_name in self.nearby_objects:
            del self.nearby_objects[obj_name]
            print(f"   {SciFiPrinter.ENV_COLOR}[环境] {obj_name} 消失了")
    
    def auto_generate_food(self):
        """自动生成食物，防止饿死"""
        # 每隔一段时间自动生成食物
        if len([obj for obj, attrs in self.nearby_objects.items() 
                if isinstance(attrs, list) and "可食用" in attrs]) < 2:
            foods = [
                ("野果", ["可食用", "水果", "自然"]),
                ("蘑菇", ["可食用", "菌类", "营养"]),
                ("坚果", ["可食用", "坚果", "能量"]),
                ("面包", ["可食用", "食物", "饱腹"])
            ]
            
            food_name, food_attrs = random.choice(foods)
            # 避免重复
            if food_name not in self.nearby_objects:
                self.nearby_objects[food_name] = food_attrs
                print(f"   {SciFiPrinter.ENV_COLOR}[环境] 自然生成了新的食物：{food_name}")
    
    def reset(self):
        """重置环境，开始新的迭代"""
        self.iteration_count += 1
        print(f"\n{'='*60}")
        SciFiPrinter.print_system(f"开始第 {self.iteration_count} 次迭代")
        print('='*60)
        
        # 保留部分物体，重置位置
        self.nearby_objects = {
            "苹果": ["可食用", "水果", "固体"],
            "石头": ["坚硬", "固体"],
            "水": ["液体", "可饮用"]
        }
        self.current_location = "起始点"

# ================= LLM教师 =================
class LLMTeacher:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    
    def extract_rule_from_surprise(self, context: Dict) -> Optional[Dict]:
        """从意外中提取规则"""
        prompt = f"""
作为世界规则分析专家，请从以下意外事件中提取因果关系：

事件背景：
- 世界状态：{context['state']}
- 物体属性：{context['obj_attrs']}
- 执行动作：{context['action']}
- 预测结果：{context['predicted']}
- 实际结果：{context['actual']}

请提取一个规则，格式如下（严格JSON格式）：
{{
    "rule_id": "R_类别_动作_属性",
    "conditions": {{
        "动作": "动作名",
        "物体属性": "触发属性"
    }},
    "effects": {{
        "状态变化": "消失/变化/产出",
        "奖励": 0.5,
        "描述": "结果描述（使用{{物体}}占位）"
    }},
    "explanation": "规则解释"
}}

只输出JSON，不要有其他文字。
"""
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是规则提取专家，只输出合法的JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # 清理可能的Markdown格式
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # 尝试解析JSON
            rule_data = json.loads(content)
            
            # 验证必要字段
            required_fields = ["rule_id", "conditions", "effects"]
            for field in required_fields:
                if field not in rule_data:
                    raise ValueError(f"缺少必要字段: {field}")
            
            print(f"   {SciFiPrinter.LEARN_COLOR}[LLM教师] 成功提取规则：{rule_data.get('rule_id')}")
            return rule_data
            
        except json.JSONDecodeError as e:
            print(f"   {SciFiPrinter.LEARN_COLOR}[LLM教师] JSON解析失败: {e}")
            return None
        except Exception as e:
            print(f"   {SciFiPrinter.LEARN_COLOR}[LLM教师] 规则提取失败: {e}")
            return None

    def generate_exploration_topic(self, known_topics: List[str]) -> str:
        """生成探索主题"""
        prompt = f"""
已知探索区域：{known_topics}

请建议一个新的探索主题，只要主题名称，不要有其他解释。
建议方向：自动化控制、机械工程、太空探索、机器人技术、行星科学、工业制造、能源系统。
先从使用火（低级需求）再到超级智能机械（高级科技）
"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=30
            )
            return response.choices[0].message.content.strip()
        except:
            return random.choice(["自然风景", "古代文明", "现代科技", "艺术世界", "食物天地"])

# ================= 决策系统 =================
class DecisionSystem:
    def __init__(self, symbolic_model: SymbolicWorldModel, inner_voice: InnerVoiceSystem, text_model: TextWorldModel):
        self.symbolic_model = symbolic_model
        self.inner_voice = inner_voice
        self.text_model = text_model
        self.action_pool = ["观察", "吃", "触摸", "探索", "休息"]

    def decide(self, nearby_objects: Dict, internal_state: Dict) -> Tuple[str, str]:
        """决策过程，包含内心独白生成"""
        SciFiPrinter.print_decision("开始思考下一步...")
        
        # 优先处理生存需求
        if internal_state["饥饿"] > 0.7:
            print(f"   {SciFiPrinter.DECISION_COLOR}饥饿感过高({internal_state['饥饿']:.1f})，必须找食物！")
            for obj_name, attrs in nearby_objects.items():
                # 兼容两种数据格式
                attr_list = attrs if isinstance(attrs, list) else attrs.get("属性", [])
                if "可食用" in attr_list:
                    decision = (obj_name, "吃")
                    thought = self.inner_voice.generate_thought(
                        self.text_model.get_world_state_text(),
                        internal_state,
                        nearby_objects,
                        decision
                    )
                    SciFiPrinter.print_thought(thought)
                    return decision
            return ("未知区域", "探索")
        
        # 好奇心驱动
        if internal_state["好奇"] > 0.8:
            print(f"   {SciFiPrinter.DECISION_COLOR}好奇心驱使，想要探索新事物")
            decision = ("未知区域", "探索")
            thought = self.inner_voice.generate_thought(
                self.text_model.get_world_state_text(),
                internal_state,
                nearby_objects,
                decision
            )
            SciFiPrinter.print_thought(thought)
            return decision
        
        # 模拟决策过程
        best_action = None
        best_reward = -999
        best_target = None
        
        for obj in nearby_objects.keys():
            for action in self.action_pool:
                prediction = self.symbolic_model.predict(
                    self.text_model.get_world_state_text(),
                    action, obj, nearby_objects
                )
                
                if prediction["success"]:
                    reward = prediction.get("reward", 0.0)
                    if action not in internal_state.get("tried_actions", []):
                        reward += 0.2  # 鼓励尝试新动作
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_action = action
                        best_target = obj
        
        if best_action:
            decision = (best_target, best_action)
            print(f"   {SciFiPrinter.DECISION_COLOR}决策完成：对[{best_target}]执行[{best_action}]，预期奖励:{best_reward:.2f}")
            
            # 生成内心独白
            thought = self.inner_voice.generate_thought(
                self.text_model.get_world_state_text(),
                internal_state,
                nearby_objects,
                decision
            )
            SciFiPrinter.print_thought(thought)
            
            return decision
        
        # 默认行为
        if nearby_objects:
            obj = random.choice(list(nearby_objects.keys()))
            return (obj, "观察")
        return ("周围", "观察")

# ================= AGI智能体 =================
class AGIAgent:
    def __init__(self):
        print("\n" + "="*60)
        SciFiPrinter.print_system("天依AGI 3.2 - 预测思维版（修复版）启动中…")
        print(f"{SciFiPrinter.SYSTEM_COLOR} 理念：世界是文本，我才是大脑，LLM是我的思维和老师")
        print("="*60)

        # 初始化各组件
        self.environment = BaiduTextWorld()
        self.symbolic_model = SymbolicWorldModel()
        self.text_model = TextWorldModel()
        self.teacher = LLMTeacher(DEEPSEEK_API_KEY)
        self.inner_voice = InnerVoiceSystem(DEEPSEEK_API_KEY)
        self.decision_system = DecisionSystem(self.symbolic_model, self.inner_voice, self.text_model)
        
        # 内在状态
        self.internal_state = {
            "饥饿": 0.0,
            "好奇": 0.8,
            "能量": 1.0,
            "恐惧": 0.0,
            "tried_actions": []
        }
        
        # 经验记录
        self.life_experience = []
        self.surprises = []
        self.total_iterations = 0  # 总迭代次数
        
        self._init_world()

    def _init_world(self):
        """初始化世界，统一使用列表格式存储属性"""
        # 统一使用列表格式，避免数据结构不一致
        self.environment.nearby_objects = {
            "苹果": ["可食用", "水果", "固体"],
            "石头": ["坚硬", "固体"],
            "水": ["液体", "可饮用"]
        }
        self.environment.current_location = "起始点"
        
        SciFiPrinter.print_system("世界初始化完成")
        print(f"   {SciFiPrinter.SYSTEM_COLOR}周围物体: {list(self.environment.nearby_objects.keys())}")
        print(f"   {SciFiPrinter.SYSTEM_COLOR}世界描述: {self.text_model.get_world_state_text()[:100]}...")

    def perceive(self) -> Dict:
        """感知环境"""
        return self.environment.get_nearby_objects()

    def act(self, target: str, action: str) -> Dict:
        """执行动作"""
        SciFiPrinter.print_action(f"对[{target}]执行[{action}]")
        
        # 使用符号模型预测
        prediction = self.symbolic_model.predict(
            self.text_model.get_world_state_text(),
            action, target, self.environment.nearby_objects
        )
        
        # 执行动作并获取实际结果
        actual_result = self._execute_action(target, action)
        
        # 判断是否意外
        is_surprise = not prediction["success"] or prediction.get("desc") != actual_result.get("desc")
        
        # 记录经验
        experience = {
            "target": target,
            "action": action,
            "predicted": prediction.get("desc", "未知"),
            "actual": actual_result,
            "is_surprise": is_surprise
        }
        self.life_experience.append(experience)
        
        # 如果出现意外，学习新规则
        if is_surprise and action != "探索":
            SciFiPrinter.print_surprise("这和我想的不一样！")
            self.surprises.append(experience)
            
            # 获取物体属性
            obj_attrs = self.environment.nearby_objects.get(target, [])
            
            rule_data = self.teacher.extract_rule_from_surprise({
                "state": self.text_model.get_world_state_text(),
                "obj_attrs": str(obj_attrs),
                "action": action,
                "predicted": prediction.get("desc", "未知"),
                "actual": actual_result.get("desc", "未知")
            })
            
            if rule_data:
                # 添加新规则
                new_rule = SymbolicRule(
                    id=rule_data["rule_id"],
                    conditions=rule_data["conditions"],
                    effects=rule_data["effects"],
                    confidence=0.8,
                    source="LLM学习"
                )
                self.symbolic_model.rules.append(new_rule)
                SciFiPrinter.print_learn(f"我学会了新规则：{new_rule.id}")
                
                # 立即保存知识库
                self.symbolic_model.save_knowledge()
        
        # 更新世界文本
        if actual_result.get("success"):
            self.text_model.update_world_text(action, target, actual_result.get("desc", ""))
            
            # 如果物体消失，从环境中移除
            if "消失" in actual_result.get("effects", ""):
                self.environment.remove_object(target)
        
        return actual_result

    def _execute_action(self, target: str, action: str) -> Dict:
        """实际执行动作"""
        # 探索新区域
        if target == "未知区域" and action == "探索":
            new_topic = self.teacher.generate_exploration_topic(list(self.environment.visited_topics))
            res = self.environment.explore_topic(new_topic)
            return {
                "success": True,
                "desc": f"探索了新区域：{new_topic}",
                "reward": 0.3,
                "effects": "发现新物体"
            }
        
        # 检查目标是否存在
        if target not in self.environment.nearby_objects:
            return {"success": False, "desc": f"这里没有{target}", "reward": 0.0}
        
        # 获取物体属性
        obj_attrs = self.environment.nearby_objects[target]
        if isinstance(obj_attrs, dict):
            obj_attrs = obj_attrs.get("属性", [])
        
        # 执行动作
        if action == "吃" and "可食用" in obj_attrs:
            return {"success": True, "desc": f"你吃掉了{target}，真好吃！", "reward": 0.5, "effects": "消失"}
        elif action == "观察":
            return {"success": True, "desc": f"你观察了{target}，属性：{obj_attrs}", "reward": 0.1}
        elif action == "触摸" and "热" in obj_attrs:
            return {"success": True, "desc": f"你被{target}烫伤了！", "reward": -0.5, "effects": "受伤"}
        else:
            return {"success": False, "desc": f"对{target}执行{action}无效果", "reward": 0.0}

    def update_internal_state(self, action_result: Dict):
        """更新内在状态"""
        self.internal_state["饥饿"] = min(1.0, self.internal_state["饥饿"] + 0.08)
        self.internal_state["能量"] = max(0.0, self.internal_state["能量"] - 0.05)
        
        reward = action_result.get("reward", 0.0)
        if reward > 0:
            self.internal_state["好奇"] = min(1.0, self.internal_state["好奇"] + 0.1)
        if reward < 0:
            self.internal_state["恐惧"] = min(1.0, self.internal_state["恐惧"] + 0.2)
        if "吃" in action_result.get("desc", ""):
            self.internal_state["饥饿"] = max(0.0, self.internal_state["饥饿"] - 0.5)

    def reset_for_new_iteration(self):
        """重置状态，开始新的迭代"""
        self.total_iterations += 1
        
        print(f"\n{'='*60}")
        SciFiPrinter.print_system(f"智能体完成一轮生命周期，开始第 {self.total_iterations} 次迭代")
        print('='*60)
        
        # 重置内在状态
        self.internal_state = {
            "饥饿": 0.0,
            "好奇": 0.8,
            "能量": 1.0,
            "恐惧": 0.0,
            "tried_actions": []
        }
        
        # 重置环境
        self.environment.reset()
        
        # 重置世界文本
        self.text_model.reset()
        
        # 保留学习经验（可以选择清空）
        # self.life_experience = []
        # self.surprises = []

    def live(self, cycles_per_iteration: int = 15, infinite: bool = True):
        """生命循环 - 支持无限迭代"""
        SciFiPrinter.print_system("开始生活，目标：生存并探索世界")
        print("="*60)
        
        iteration_count = 0
        
        while True:  # 无限循环
            iteration_count += 1
            
            for i in range(cycles_per_iteration):
                print(f"\n{'='*20} 第{iteration_count}次迭代 - {i+1}轮 {'='*20}")
                
                # 自动生成食物，防止饿死
                self.environment.auto_generate_food()
                
                # 感知
                nearby_objects = self.perceive()
                print(f"{SciFiPrinter.ENV_COLOR}当前位置: {self.environment.current_location}")
                print(f"   {SciFiPrinter.ENV_COLOR}周围物体: {list(nearby_objects.keys())}")
                print(f"{SciFiPrinter.SYSTEM_COLOR}内在状态: 饥饿({self.internal_state['饥饿']:.1f}) 好奇({self.internal_state['好奇']:.1f}) 恐惧({self.internal_state['恐惧']:.1f})")
                
                # 决策
                target, action = self.decision_system.decide(nearby_objects, self.internal_state)
                
                # 记录尝试过的动作
                if action not in self.internal_state["tried_actions"]:
                    self.internal_state["tried_actions"].append(action)
                
                # 行动
                result = self.act(target, action)
                print(f"{SciFiPrinter.ACTION_COLOR}结果: {result.get('desc', '无描述')}")
                
                # 更新内在状态
                self.update_internal_state(result)
                
                # 检查迭代条件（不再退出，而是重置）
                if self.internal_state["饥饿"] >= 1.0:
                    print(f"\n{SciFiPrinter.SYSTEM_COLOR}[迭代] 能量耗尽，进入下一个迭代周期...")
                    break
                if self.internal_state["能量"] <= 0.0:
                    print(f"\n{SciFiPrinter.SYSTEM_COLOR}[休息] 能量耗尽，短暂休息...")
                    self.internal_state["能量"] = 0.5
                
                time.sleep(0.3)
            
            # 完成一次迭代后的处理
            self._iteration_summary(iteration_count)
            
            # 重置状态，开始新的迭代
            self.reset_for_new_iteration()
            
            # 如果不是无限模式，则退出
            if not infinite:
                break
            
            # 短暂休息后继续
            SciFiPrinter.print_system("3秒后开始下一轮迭代...")
            time.sleep(3)

    def _iteration_summary(self, iteration_num: int):
        """迭代总结"""
        print("\n" + "="*60)
        SciFiPrinter.print_summary(f"迭代 {iteration_num} 完成")
        print("="*60)
        print(f"\n{SciFiPrinter.SUMMARY_COLOR}本迭代总结:")
        print(f"   - {SciFiPrinter.SUMMARY_COLOR}总经历: {len(self.life_experience)} 次")
        print(f"   - {SciFiPrinter.SUMMARY_COLOR}意外事件: {len(self.surprises)} 次")
        print(f"   - {SciFiPrinter.SUMMARY_COLOR}学会规则: {len(self.symbolic_model.rules)} 条")
        print(f"   - {SciFiPrinter.SUMMARY_COLOR}探索区域: {len(self.environment.visited_topics)} 个")
        
        print(f"\n{SciFiPrinter.KNOWLEDGE_COLOR}知识库状态:")
        learned_rules = [r for r in self.symbolic_model.rules if r.source == "LLM学习"]
        print(f"   - {SciFiPrinter.KNOWLEDGE_COLOR}已学习规则: {len(learned_rules)} 条")
        
        # 保存知识库
        if learned_rules:
            self.symbolic_model.save_knowledge()

# ================= 主程序 =================
if __name__ == "__main__":
    try:
        agent = AGIAgent()
        agent.live(cycles_per_iteration=15, infinite=True)  # 无限循环运行
    except KeyboardInterrupt:
        print(f"\n\n{SciFiPrinter.SYSTEM_COLOR}[系统] 用户中断，正在保存知识库…")
        agent.symbolic_model.save_knowledge()
        SciFiPrinter.print_system("知识库已保存，程序退出")
    except Exception as e:
        print(f"{SciFiPrinter.SYSTEM_COLOR}程序异常: {e}")
        traceback.print_exc()
        # 即使异常也尝试保存知识库
        if 'agent' in locals():
            agent.symbolic_model.save_knowledge()