import streamlit as st
from pathlib import Path
from src.pipeline import Pipeline, max_config
from src.questions_processing import QuestionsProcessor
import json

# 你可以让 root_path 固定，也可以让用户输入
root_path = Path("data/stock_data")
pipeline = Pipeline(root_path, run_config=max_config)

st.set_page_config(page_title="RAG Challenge 2", layout="wide")

# 页面标题
st.markdown("""
<div style='background: linear-gradient(90deg, #7b2ff2 0%, #f357a8 100%); padding: 20px 0; border-radius: 12px; text-align: center;'>
    <h2 style='color: white; margin: 0;'>🚀 RAG Challenge 2</h2>
    <div style='color: #fff; font-size: 16px;'>基于深度RAG系统，由RTX 5080 GPU加速 | 支持多公司年报问答 | 向量检索+LLM推理+通义千问（qwen-turbo）</div>
</div>
""", unsafe_allow_html=True)

# 左侧输入区
with st.sidebar:
    st.header("查询设置")
    # 仅单问题输入
    user_question = st.text_area("输入问题", "请简要总结中芯国际2022年主营业务的主要内容。", height=80)
    st.caption("提示：知识库当前仅收录「中芯国际」，问题中必须包含公司名才能检索")
    submit_btn = st.button("生成答案", use_container_width=True)

# 右侧主内容区
st.markdown("<h3 style='margin-top: 24px;'>检索结果</h3>", unsafe_allow_html=True)

if submit_btn and user_question.strip():
    with st.spinner("正在生成答案，请稍候..."):
        try:
            answer = pipeline.answer_single_question(user_question, kind="string")
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