import streamlit as st
from pathlib import Path
from src.pipeline import Pipeline, max_config
from src.questions_processing import QuestionsProcessor
from src.env_loader import generation_model
import json

# 你可以让 root_path 固定，也可以让用户输入
root_path = Path("data/stock_data")
pipeline = Pipeline(root_path, run_config=max_config)

# 模型名实时取自 env_loader，不再硬编码在文案里 ——
# 否则换 GENERATION_MODEL 后文案会静默过期（原先写死 qwen-turbo，实际已换 qwen-plus）。
MODEL_NAME = generation_model()

st.set_page_config(page_title="中文金融报告 RAG 问答", layout="wide")

# 页面标题
st.markdown(f"""
<div style='background: linear-gradient(90deg, #7b2ff2 0%, #f357a8 100%); padding: 20px 0; border-radius: 12px; text-align: center;'>
    <h2 style='color: white; margin: 0;'>🚀 中文金融报告 RAG 问答</h2>
    <div style='color: #fff; font-size: 16px;'>多文档向量检索 + LLM 重排 + 父文档检索 | 引文式页码溯源 | 生成模型 {MODEL_NAME}</div>
</div>
""", unsafe_allow_html=True)

# 左侧输入区
with st.sidebar:
    st.header("查询设置")
    # 仅单问题输入
    user_question = st.text_area("输入问题", "中芯国际2024年的营收和利润情况如何？", height=80)
    kind = st.selectbox(
        "答案类型",
        options=["string", "number", "boolean", "name", "names"],
        index=0,
        help=("对应 prompts 里的不同 Schema。\n"
              "string=自由文本；number=数值（启用币种/单位校验与禁止推导的防幻觉规则）；\n"
              "boolean=是否类；name=单一实体名；names=实体列表。\n"
              "默认 string，与批量流程 questions.json 中各题的 kind 一致。"),
    )
    st.caption("提示：知识库当前仅收录「中芯国际」，问题中必须包含公司名才能检索")
    submit_btn = st.button("生成答案", use_container_width=True)

# 右侧主内容区
st.markdown("<h3 style='margin-top: 24px;'>检索结果</h3>", unsafe_allow_html=True)

if submit_btn and user_question.strip():
    with st.spinner("正在生成答案，请稍候..."):
        try:
            answer = pipeline.answer_single_question(user_question, kind=kind)
            # 兼容 answer 可能为 str 或 dict
            if isinstance(answer, str):
                try:
                    answer_dict = json.loads(answer)
                except Exception:
                    st.error("返回内容无法解析为结构化答案：" + str(answer))
                    answer_dict = {}
            else:
                answer_dict = answer
            # 直接取返回结果中的顶层字段
            # 实际返回结构：step_by_step_analysis / reasoning_summary /
            #              relevant_quotes / relevant_pages / final_answer / references
            step_by_step = answer_dict.get("step_by_step_analysis", "-")
            reasoning_summary = answer_dict.get("reasoning_summary", "-")
            relevant_quotes = answer_dict.get("relevant_quotes", [])
            relevant_pages = answer_dict.get("relevant_pages", [])
            final_answer = answer_dict.get("final_answer", "-")
            references = answer_dict.get("references", [])
            # 打印调试
            print("[DEBUG] step_by_step_analysis:", step_by_step)
            print("[DEBUG] reasoning_summary:", reasoning_summary)
            print("[DEBUG] relevant_quotes:", relevant_quotes)
            print("[DEBUG] relevant_pages:", relevant_pages)
            print("[DEBUG] final_answer:", final_answer)
            st.markdown("**分步推理：**")
            st.info(step_by_step)
            st.markdown("**推理摘要：**")
            st.success(reasoning_summary)
            if relevant_quotes:
                st.markdown("**引用原文：**")
                for q in relevant_quotes:
                    st.markdown(f"> {q}")
            st.markdown("**相关页面：** ")
            st.write(relevant_pages)
            if references:
                st.markdown("**引用来源：** ")
                st.write(references)
            st.markdown("**最终答案：**")
            st.markdown(f"<div style='background:#f6f8fa;padding:16px;border-radius:8px;font-size:18px;'>{final_answer}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"生成答案时出错: {e}")
else:
    st.info("请在左侧输入问题并点击【生成答案】") 