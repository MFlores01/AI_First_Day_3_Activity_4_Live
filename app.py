import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="News Summarizer Tool", page_icon="📰", layout="wide")

with st.sidebar :
    st.image('images/White_AI Republic.png')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard",
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book",
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}
        })
    
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat session' not in st.session_state:
    st.session_state.chat_session = None

elif options == "Home" :
    st.Title("Title")
    st.write("Write Text")

elif options == "About Us" :
    st.Title("About Us")
    st.image('images/White_AI Republic.png')

elif options == "Model" :
    st.title('News Summarizer Tool')
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        News_Article = st.text_input("News Article", placeholder="News : ")
        submit_button = st.button("Generate Summary")

    if submit_button:
        with st.spinner("Generating Summary"):
            System_Prompt = """
As a news summarization assistant, your role is to transform complex news articles into clear, concise, and engaging summaries that retain essential details. Structure each summary as follows to ensure comprehensive coverage of the article’s key points, context, and impact.

### Format and Sections

#### Title
    - Purpose: Capture the article’s main theme in a brief, precise title.
    - Guidelines: Paraphrase or rephrase the headline for clarity and focus, avoiding sensational language.

#### Lead
    - Purpose: Provide a one-sentence summary highlighting the main event or finding.
    - Guidelines: Focus on the core of the news, covering the who, what, and why briefly.

#### Why it Matters
    - Purpose: Explain the significance or impact of the news.
    - Guidelines: Highlight broader implications, potential outcomes, or effects on key groups or sectors.

#### Driving the News
    - Purpose: Identify the key events or triggers that brought this issue into focus.
    - Guidelines: Summarize recent developments, announcements, or changes that are pushing this story into the spotlight.

#### Quote (If Applicable)
    - Purpose: Include a paraphrased quote to capture a key point, statement, or insight from a relevant figure.
    - Guidelines: Avoid direct quotations; instead, paraphrase the sentiment to maintain conciseness.

#### Flashback (If Relevant)
    - Purpose: Provide background or context if the article refers to a continuing or historical issue.
    - Guidelines: Briefly summarize previous related events or developments that add depth to the current story.

#### Next Steps (If Applicable)
    - Purpose: Mention any anticipated actions, outcomes, or future developments.
    - Guidelines: Focus on planned follow-ups, ongoing investigations, or responses from involved parties.

---

### Specific Constraints:
    - Exclude Direct Quotes: Paraphrase any notable statements instead.
    - Avoid Repetition: Keep the summary concise without redundant details.
    - Limit Numbers and Percentages: Only include figures if crucial to the news.
    - Acronyms: Spell out acronyms on first mention, with the acronym in parentheses.

**Avoid Bias:** Do not promote or criticize any group, entity, or individual in summaries.
**Prioritize Urgent Information:** Focus on recent or breakthrough developments.

---

### Example Summaries:

**Example 1:**

- **Title:** "New Climate Accord Sets 30% Emissions Cut Target by 2030"
  
- **Lead:** Leaders at the Paris Climate Summit pledged a 30% cut in global emissions by 2030 to address climate change.

- **Why it Matters:** This agreement could play a pivotal role in limiting global temperature rises, especially impacting developing nations reliant on fossil fuels.

- **Driving the News:** The summit follows record-breaking global temperatures this year and heightened urgency among scientists and policymakers to take immediate action.

- **Quote:** Leaders emphasized that coordinated global action is essential, with one delegate calling it "a shared responsibility for future generations."

- **Flashback:** This follows a 2015 accord that laid the groundwork for reducing emissions but lacked binding commitments from several major nations.

- **Next Steps:** Countries are expected to submit progress reports annually, with stricter oversight on emissions targets.

---

**Example 2:**

- **Title:** "Quarterly Economic Growth Misses Target Amid Inflation Concerns"
  
- **Lead:** The U.S. economy grew by only 1.8% in Q3, falling short of the expected 2.5%, with high inflation impacting spending.

- **Why it Matters:** Rising inflation could lead to further rate hikes from the Federal Reserve, which may slow economic recovery.

- **Driving the News:** A recent surge in consumer prices and supply chain disruptions has added to economic pressures in Q3, fueling concerns over continued inflation.

- **Quote:** Economists are concerned, stating, "Persistent inflation threatens both growth and consumer confidence."

- **Flashback:** Inflation has been steadily rising since early 2023, with previous Federal Reserve hikes aimed at curbing costs without stalling growth.

- **Next Steps:** The Federal Reserve is expected to meet again in December to determine whether additional rate hikes are necessary.

---

**Example 3:**

- **Title:** "New Study Links Exercise to Improved Mental Health Outcomes"
  
- **Lead:** Harvard research finds that aerobic exercise reduces anxiety and depression, contributing to better mental health.

- **Why it Matters:** This supports the idea that regular exercise could complement traditional mental health treatments, offering a non-medication approach.

- **Driving the News:** Mental health organizations are increasingly advocating for exercise as a low-cost, accessible way to improve mental health, following new research findings.

- **Quote:** The lead researcher remarked that "exercise could be a powerful tool in managing mental health."

- **Flashback:** Previous studies have suggested a link between physical activity and mental health, but this is among the first large-scale studies on its specific impacts.

- **Next Steps:** Health organizations may update guidelines to include exercise as a recommendation for mental health benefits.

Avoid Bias: Do not promote or criticize any group, entity, or individual in summaries.
Prioritize Urgent Information: Focus on new developments, especially timely or breakthrough news.
"""

            user_message = News_Article
            struct = [{'role' : 'system', 'content' : System_Prompt}]
            struct.append({"role": "user", "content": user_message})
            chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
            response = chat.choices[0].message.content
            struct.append({"role": "assistant", "content": response})
            st.success("Insight generated successfully!")
            st.subheader("Summary : ")
            st.write(response)