import streamlit as st
import time
from dotenv import load_dotenv
import os
from agent import build_search_agent, build_scraping_agent, build_writer_chain, build_critic_chain
from pipeline import get_tool_output

# Set page configuration first
st.set_page_config(
    page_title="Regent AI Research Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Inject Custom CSS for premium styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@300;400;500;600;700&display=swap');

body {
    font-family: 'Inter', sans-serif;
}

.title-container {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
    padding: 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.title-container h1 {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    margin: 0;
    font-size: 2.5rem;
    letter-spacing: -0.025em;
}

.title-container p {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
    font-weight: 300;
}

.card {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.status-badge {
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 600;
}

.status-pending { background-color: #E2E8F0; color: #475569; }
.status-running { background-color: #DBEAFE; color: #1E40AF; }
.status-success { background-color: #D1FAE5; color: #065F46; }

</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/graduation-cap.png", width=100)
    st.title("Settings")
    
    # Check for API Keys in environment
    mistral_key = os.getenv("MISTRAL_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    st.markdown("### API Key Status")
    if mistral_key:
        st.success("✅ Mistral API Key loaded")
    else:
        st.warning("⚠️ Mistral API Key missing in .env")
        mistral_input = st.text_input("Enter Mistral API Key:", type="password")
        if mistral_input:
            os.environ["MISTRAL_API_KEY"] = mistral_input
            st.success("Mistral Key set!")
            
    if tavily_key:
        st.success("✅ Tavily API Key loaded")
    else:
        st.warning("⚠️ Tavily API Key missing in .env")
        tavily_input = st.text_input("Enter Tavily API Key:", type="password")
        if tavily_input:
            os.environ["TAVILY_API_KEY"] = tavily_input
            st.success("Tavily Key set!")
            
    st.markdown("---")
    st.markdown("### About Regent")
    st.info(
        "Regent is an AI-powered Multi-Agent Research Assistant. It combines Web Search, Scrape URL, Writing, and Critic agents to produce comprehensive, peer-reviewed reports."
    )

# Main Title Header
st.markdown("""
<div class="title-container">
    <h1>🎓 Regent Research Agent</h1>
    <p>Enter a topic below to initiate the autonomous research, writing, and review pipeline.</p>
</div>
""", unsafe_allow_html=True)

# Main UI layout
col_input, col_run = st.columns([4, 1])

with col_input:
    topic = st.text_input(
        "Enter Research Topic:",
        placeholder="e.g. Impact of war on oil prices",
        label_visibility="collapsed"
    )

with col_run:
    run_btn = st.button("Generate Report", type="primary", use_container_width=True)

# Session state initialization
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "pipeline_completed" not in st.session_state:
    st.session_state.pipeline_completed = False
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = {}

def run_pipeline_stream(topic: str):
    state = {}
    
    # Step 1: Search Agent
    yield 1, "Searching the web for information...", state
    search_agent = build_search_agent()
    search_result = search_agent.invoke(
        {"messages" : [{"role" : "user", "content" : f"Search the web for information on {topic}"}]}
    )
    state["search_result"] = get_tool_output(search_result["messages"], "web_search")
    
    # Pause between API requests to prevent 429
    time.sleep(5)
    
    # Step 2: Scraping Agent
    yield 2, "Extracting deep content from the most relevant source...", state
    scraping_agent = build_scraping_agent()
    scraping_result = scraping_agent.invoke({
        "messages": [{"role" : "user", "content" : f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_result'][:800]}"}]}
    )
    state['scraped_content'] = get_tool_output(scraping_result["messages"], "scrape_url")
    
    time.sleep(5)
    
    # Step 3: Writer Chain
    yield 3, "Drafting the research report...", state
    research_combined = (
        f"SEARCH RESULTS : \n {state['search_result']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )
    state["report"] = build_writer_chain().invoke({
        "topic" : topic,
        "research" : research_combined
    })
    
    time.sleep(5)
    
    # Step 4: Critic Chain
    yield 4, "Reviewing draft and scoring...", state
    state["feedback"] = build_critic_chain().invoke({
        "report": state['report']
    })
    
    yield 5, "Done", state


if run_btn and topic:
    # Check for keys before starting
    if not os.getenv("MISTRAL_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error("Please ensure both Mistral and Tavily API keys are set (in your .env or the sidebar).")
    else:
        st.session_state.pipeline_running = True
        st.session_state.pipeline_completed = False
        st.session_state.pipeline_state = {}
        
        # Display progress UI placeholder
        progress_placeholder = st.empty()
        
        # Create tabs/layout for results
        results_container = st.container()
        
        # Run pipeline and update stream
        for step, status, current_state in run_pipeline_stream(topic):
            
            with progress_placeholder.container():
                st.markdown(f"### ⚙️ Pipeline Progress")
                
                # Render step-by-step progress cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if step == 1:
                        st.info("🔍 Searching...")
                    elif step > 1:
                        st.success("🔍 Search Complete")
                    else:
                        st.text("🔍 Step 1: Search")
                        
                with col2:
                    if step == 2:
                        st.info("📄 Scraping...")
                    elif step > 2:
                        st.success("📄 Scraping Complete")
                    else:
                        st.text("📄 Step 2: Scrape")
                        
                with col3:
                    if step == 3:
                        st.info("✍️ Writing...")
                    elif step > 3:
                        st.success("✍️ Writing Complete")
                    else:
                        st.text("✍️ Step 3: Write")
                        
                with col4:
                    if step == 4:
                        st.info("⚖️ Reviewing...")
                    elif step > 4:
                        st.success("⚖️ Review Complete")
                    else:
                        st.text("⚖️ Step 4: Review")
                
                if step < 5:
                    st.write(f"**Current Status:** {status}")
                    st.progress(step * 20)
                else:
                    st.success("🎉 Research Pipeline Completed Successfully!")
                    st.progress(100)
            
            # Save state in session state
            st.session_state.pipeline_state = current_state
            
        st.session_state.pipeline_running = False
        st.session_state.pipeline_completed = True

# Display results once completed
if st.session_state.pipeline_completed:
    state = st.session_state.pipeline_state
    
    st.markdown("---")
    
    # Left column: Report & Critic, Right column: Pipeline outputs
    col_report, col_logs = st.columns([3, 2])
    
    with col_report:
        st.markdown("## 📄 Research Report")
        
        # Display report in a styled block
        st.markdown(f"""
        <div style="background-color: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 2rem;">
            {state.get('report', 'No report generated.')}
        </div>
        """, unsafe_allow_html=True)
        
        # Download button
        st.download_button(
            label="⬇️ Download Markdown Report",
            data=state.get('report', ''),
            file_name=f"{topic.replace(' ', '_')}_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col_logs:
        st.markdown("## 🔍 Review & Logs")
        
        # Critic feedback
        with st.expander("⚖️ Critic's Review Feedback", expanded=True):
            st.markdown(state.get('feedback', 'No feedback.'))
            
        # Search Results
        with st.expander("🔍 Step 1: Web Search Results", expanded=False):
            st.code(state.get('search_result', 'No search results found.'))
            
        # Scraped Content
        with st.expander("📄 Step 2: Scraped Source Content", expanded=False):
            st.text_area("Scraped Text", state.get('scraped_content', 'No content scraped.'), height=300)
