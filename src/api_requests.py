import os
from typing import List, Dict, Optional, Literal
from openai import OpenAI
import src.prompts as prompts
from src.structured_output import parse_structured_output, build_schema_instruction
from tenacity import (retry, stop_after_attempt, wait_exponential,
                        retry_if_exception_type)
import dashscope
from src.env_loader import load_project_env, generation_model
from src.dashscope_errors import (
    DashScopeError,
    DashScopeThrottled,
    raise_if_api_error,
)

# 结构化答案必须具备的字段。各 schema 的字段名一致，只有 relevant_quotes 是
# 引文式引用后新增的；relevant_pages 仍在（由引文解析回填）。
REQUIRED_ANSWER_FIELDS = ("step_by_step_analysis", "reasoning_summary", "relevant_quotes")

# OpenAI基础处理器，封装了消息发送、结构化输出、计费等逻辑
class BaseOpenaiProcessor:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.default_model = 'gpt-4o-2024-08-06'
        # self.default_model = 'gpt-4o-mini-2024-07-18',

    def set_up_llm(self):
        # 加载OpenAI API密钥，初始化LLM
        load_project_env()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None, # For deterministic ouptputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None
        ):
        # 发送消息到OpenAI，支持结构化/非结构化输出
        if model is None:
            model = self.default_model
        params = {
            "model": model,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content}
            ]
        }
        
        # 部分模型不支持temperature
        if "o3-mini" not in model:
            params["temperature"] = temperature
            
        if not is_structured:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content

        elif is_structured:
            params["response_format"] = response_format
            completion = self.llm.beta.chat.completions.parse(**params)

            response = completion.choices[0].message.parsed
            content = response.dict()

        self.response_data = {"model": completion.model, "input_tokens": completion.usage.prompt_tokens, "output_tokens": completion.usage.completion_tokens}
        print(self.response_data)

        return content


class APIProcessor:
    """按 provider 路由到对应的处理器实现。

    原本还有 ``ibm`` 与 ``gemini`` 两个分支，但全项目没有任何地方把
    ``RunConfig.api_provider`` 设成这两个值 —— 它们只能由外部代码直接
    构造 ``APIProcessor(provider=...)`` 才会走到，属于不可达分支，已删除。
    需要恢复时按当时的形态补回对应处理器即可。
    """

    SUPPORTED_PROVIDERS = ("openai", "dashscope")

    def __init__(self, provider: Literal["openai", "dashscope"] = "dashscope"):
        self.provider = provider.lower()
        if self.provider == "openai":
            self.processor = BaseOpenaiProcessor()
        elif self.provider == "dashscope":
            self.processor = BaseDashscopeProcessor()
        else:
            raise ValueError(
                f"不支持的 api_provider: {self.provider}。"
                f"可选值：{', '.join(self.SUPPORTED_PROVIDERS)}"
            )

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        Routes the send_message call to the appropriate processor.
        The underlying processor's send_message method is responsible for handling the parameters.
        """
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs
        )

    def get_answer_from_rag_context(self, question, rag_context, schema, model):
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)
        
        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format
        )
        self.response_data = self.processor.response_data
        
        # 各 provider 的 send_message 现在都会返回经 Pydantic 校验的结构，或显式的
        # 降级记录。此处只做最终一致性检查。
        #
        # 历史包袱：原先这里有一大段启发式兜底，会把 final_answer 里"看起来像
        # JSON 的字符串"再解析一遍塞回答案。那段代码正是 answers_qwen_turbo.json
        # 里 4 条 value 变成整坨 JSON 的直接成因，而且退出码为 0、无任何告警。
        # 解析职责已上移到 src/structured_output.py，这里不再重复处理。
        if not isinstance(answer_dict, dict):
            print(f"Warning: provider '{self.provider}' returned "
                  f"{type(answer_dict).__name__}, not a dict; degrading.")
            answer_dict = {
                "step_by_step_analysis": "",
                "reasoning_summary": "",
                "relevant_quotes": [],
                "relevant_pages": [],
                "final_answer": "N/A",
                "_degraded": True,
                "_parse_errors": [f"unexpected type {type(answer_dict).__name__}"],
            }
        elif answer_dict.get("_degraded"):
            # 保留 send_message 标注的降级状态，不要掩盖
            print("Warning: structured output was degraded upstream; "
                  "this answer must not be treated as a valid model output.")
        else:
            missing = [f for f in REQUIRED_ANSWER_FIELDS if f not in answer_dict]
            if missing:
                print(f"Warning: answer is missing expected fields {missing}; degrading.")
                answer_dict = {
                    "step_by_step_analysis": answer_dict.get("step_by_step_analysis", ""),
                    "reasoning_summary": answer_dict.get("reasoning_summary", ""),
                    "relevant_quotes": answer_dict.get("relevant_quotes", []),
                    "relevant_pages": answer_dict.get("relevant_pages", []),
                    "final_answer": answer_dict.get("final_answer", "N/A"),
                    "_degraded": True,
                    "_parse_errors": [f"missing fields {missing}"],
                }
        return answer_dict


    def _build_rag_context_prompts(self, schema):
        """Return prompts tuple for the given schema."""
        # 之前只有 ibm / gemini 会把 Pydantic schema 注入 system prompt，
        # dashscope 与 openai 被排除在外 —— 而 dashscope 是默认 provider，
        # 于是模型连字段名都只能从示例里猜。现在全 provider 都注入。
        use_schema_prompt = True
        
        if schema == "name":
            system_prompt = (prompts.AnswerWithRAGContextNamePrompt.system_prompt_with_schema 
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamePrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamePrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamePrompt.user_prompt
        elif schema == "number":
            system_prompt = (prompts.AnswerWithRAGContextNumberPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNumberPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNumberPrompt.user_prompt
        elif schema == "boolean":
            system_prompt = (prompts.AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextBooleanPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextBooleanPrompt.user_prompt
        elif schema == "names":
            system_prompt = (prompts.AnswerWithRAGContextNamesPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamesPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamesPrompt.user_prompt
        elif schema == "comparative":
            system_prompt = (prompts.ComparativeAnswerPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.ComparativeAnswerPrompt.system_prompt)
            response_format = prompts.ComparativeAnswerPrompt.AnswerSchema
            user_prompt = prompts.ComparativeAnswerPrompt.user_prompt
        elif schema == "string":
            # 新增：支持开放性文本问题
            system_prompt = (prompts.AnswerWithRAGContextStringPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextStringPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextStringPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextStringPrompt.user_prompt
        else:
            raise ValueError(f"Unsupported schema: {schema}")
        return system_prompt, response_format, user_prompt

    def get_rephrased_questions(self, original_question: str, companies: List[str]) -> Dict[str, str]:
        """Use LLM to break down a comparative question into individual questions."""
        answer_dict = self.processor.send_message(
            system_content=prompts.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies])
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions
        )
        
        # Convert the answer_dict to the desired format
        questions_dict = {item["company_name"]: item["question"] for item in answer_dict["questions"]}
        
        return questions_dict


# DashScope基础处理器，支持Qwen大模型对话
def _extract_content(response) -> Optional[str]:
    """从 DashScope 响应里取文本内容，取不到返回 None。

    ``dashscope`` 返回的是 ``DictWrapper``，既支持 ``resp['output']`` 也支持
    ``resp.output``，但 ``hasattr`` 对 dict 形式的键探测并不可靠。这里两种
    取法都试，取不到就明确返回 None，由调用方抛错 —— 绝不把整个响应
    ``str()`` 之后当成模型正文，那会让错误信息被当成答案继续流下去。
    """
    output = None
    if isinstance(response, dict):
        output = response.get("output")
    else:
        output = getattr(response, "output", None)

    if not output:
        return None

    choices = output.get("choices") if isinstance(output, dict) else getattr(output, "choices", None)
    if not choices:
        return None

    message = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
    if not message:
        return None

    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


class BaseDashscopeProcessor:
    def __init__(self):
        # 从环境变量读取API-KEY
        load_project_env()
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.default_model = generation_model()

    @staticmethod
    @retry(wait=wait_exponential(multiplier=3, min=3, max=40),
           stop=stop_after_attempt(4),
           retry=retry_if_exception_type(DashScopeThrottled),
           reraise=True)
    def _call_generation(**kwargs):
        """统一生成调用入口，只对限流退避重试。"""
        return dashscope.Generation.call(**kwargs)

    def send_message(
        self,
        model=None,
        temperature=0.1,
        seed=None,  # 兼容参数，暂不使用
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        发送消息到 DashScope Qwen 大模型。

        结构化输出：qwen 系列不支持 OpenAI 式的 response_format 强约束，但支持
        ``response_format={'type': 'json_object'}`` JSON 模式。开启后仍可能返回
        带围栏 / 带说明文字 / 字段嵌套错乱的文本，因此还要过
        :func:`src.structured_output.parse_structured_output` 做提取 + 校验 +
        修补，并把实际使用的解析策略记录在 ``self.last_parse_strategy``。
        """
        if model is None:
            model = self.default_model

        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if human_content:
            messages.append({"role": "user", "content": human_content})

        call_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "result_format": "message",
        }

        if is_structured and response_format is not None:
            # 让 schema 约束进入提示词：qwen 没有强约束，只能靠指令
            directive = build_schema_instruction(response_format)
            call_kwargs["messages"] = [
                {"role": "system", "content": f"{system_content}\n\n---\n\n{directive}"}
                if system_content else {"role": "system", "content": directive},
            ] + ([{"role": "user", "content": human_content}] if human_content else [])
            try:
                response = self._call_generation(
                    response_format={"type": "json_object"}, **call_kwargs
                )
            except DashScopeError as exc:
                # 仅"模型/网关不支持 JSON 模式"才回退。限流/额度/参数错误必须
                # 继续抛出 —— 回退只会再失败一次，然后把限流伪装成解析失败。
                if "response_format" not in str(exc) and "json_object" not in str(exc):
                    raise
                print(f"Warning: DashScope rejected response_format=json_object ({exc}); "
                      f"falling back to plain generation.")
                response = self._call_generation(**call_kwargs)
        else:
            response = self._call_generation(**call_kwargs)

        # 失败时 response 是 dict 而 output 为 None（限流 / 额度 / 参数）。
        # 原实现用 `hasattr(response, 'output')` 探测，对 dict 恒为 False，
        # 于是 content 变成字符串 "{'status_code': 429, ...}"，随后解析失败、
        # 返回 final_answer="N/A" 的降级记录 —— **限流被伪装成"答案不可得"**，
        # 退出码 0、日志无告警，比直接报错危险得多。
        raise_if_api_error(response, f"{model} 结构化生成" if is_structured else f"{model} 生成")

        content = _extract_content(response)
        if content is None:
            raise DashScopeError(
                f"DashScope 返回的响应里没有可用的文本内容（model={model}）。\n"
                f"  原始响应 = {str(response)[:300]}"
            )

        usage = getattr(response, 'usage', None)
        self.response_data = {
            "model": model,
            "input_tokens": getattr(usage, 'input_tokens', None),
            "output_tokens": getattr(usage, 'output_tokens', None),
        }

        if not is_structured or response_format is None:
            self.last_parse_strategy = "unstructured"
            return content

        outcome = parse_structured_output(content, response_format)
        self.last_parse_strategy = outcome.strategy
        if outcome.ok:
            return outcome.data

        # 降级必须**显式可辨**，不能把原文塞进 final_answer 伪装成正常答案 ——
        # 那样退出码是 0、日志没有告警，错误会一路流进提交物。
        print(f"Warning: structured output FAILED ({outcome.summary()}); "
              f"returning a degraded record.")
        return {
            "step_by_step_analysis": "",
            "reasoning_summary": "",
            "relevant_quotes": [],
            "relevant_pages": [],
            "final_answer": "N/A",
            "_degraded": True,
            "_parse_errors": outcome.errors,
        }
