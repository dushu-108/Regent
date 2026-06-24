from agent import build_search_agent, build_scraping_agent, writer_chain, critic_chain
import time

def run_research_pipeline(topic : str) -> dict:

    state = {}

    print("\n" + " ="*50)
    print("Step 1 - Search Agent is working....")
    print(" ="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke(
        {"messages" : [{"role" : "user", "content" : f"Search the web for information on {topic}"}]}
    )

    state["search_result"] = search_result["messages"][-1].content
    print("Search Results :", state["search_result"])

    time.sleep(5)
    print("\n" + " ="*50)
    print("Step 2 - Scraping Agent is working....")

    print(" ="*50)
    
    scraping_agent = build_scraping_agent()
    scraping_result = scraping_agent.invoke({
        "messages": [{"role" : "user", "content" : f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_result'][:800]}"}]}
    )

    state['scraped_content'] = scraping_result['messages'][-1].content
    print("\nscraped content: \n", state['scraped_content'])

    time.sleep(5)
    print("\n"+" ="*50)
    print("Step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_result']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n",state['report'])

    time.sleep(5)
    print("\n"+" ="*50)
    print("Step 4 - Critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report":state['report']
    })

    print("\n critic report \n", state['feedback'])

    return state        
    
if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)


