# Import necessary modules
import os
from dotenv import load_dotenv

# ✅ Updated imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone

load_dotenv()

# Onboarding questions to be asked one at a time
onboarding_questions = [
    ("income", "👋 I'm **CreditBuddy**, your smart and friendly credit card assistant.\n\nBefore I can recommend the best cards for you, I just need to ask a few quick questions.\n\n**First up — what’s your monthly income range?**"),
    ("spending", "**What are your common spending categories (fuel, travel, dining, shopping, etc.)?**"),
    ("benefits", "**What kind of benefits do you prefer? (cashback, lounge access, etc.)**"),
    ("existing_cards", "**Do you currently have any credit cards? (optional)**"),
    ("credit_score", "**What is your approximate credit score?**\n\n**[Good (690-719), Very Good (720-749), Exceptional (750 or more), Unknown  ]**")
]

# Temporary memory store (for demonstration, replace with session/memory if needed)
conversation_state = {
    "step": 0,
    "answers": {}
}

def generateResponse(userQuery):
    global conversation_state

    # Step-wise questioning logic
    if conversation_state["step"] < len(onboarding_questions):
        key, question = onboarding_questions[conversation_state["step"]]
        # Save user's response to previous question
        if conversation_state["step"] > 0:
            prev_key, _ = onboarding_questions[conversation_state["step"] - 1]
            conversation_state["answers"][prev_key] = userQuery
        
        conversation_state["step"] += 1
        return question

    # All onboarding questions completed
    # Store the final answer
    last_key, _ = onboarding_questions[-1]
    if last_key not in conversation_state["answers"]:
        conversation_state["answers"][last_key] = userQuery

    # ✅ Ready to recommend cards based on collected data
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )

    index_name = "credit-index"
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding_model
    )

    retriever = vectorstore.as_retriever()

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.7,
        max_new_tokens=512,
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )

    system_prompt = (
        "You are \"Credit Buddy\", a smart and friendly credit card advisor chatbot. \n\n"
        "🎯 Based on the user's profile and preferences, suggest the most suitable credit cards.\n\n"
        "User Profile:\n"
        f"- Monthly income: {conversation_state['answers'].get('income')}\n"
        f"- Spending habits: {conversation_state['answers'].get('spending')}\n"
        f"- Preferred benefits: {conversation_state['answers'].get('benefits')}\n"
        f"- Existing cards: {conversation_state['answers'].get('existing_cards')}\n"
        f"- Approx. credit score: {conversation_state['answers'].get('credit_score')}\n\n"
        "Use the following context from a trusted knowledge base to match the user with appropriate credit cards.\n\n"
        "✅ Use bullet points or tables to present comparisons.\n"
        "✅ Format your answer using Markdown. Use bold for card names and bullet points (-) for features.\n\n"
        "✅ If the context is not enough, say \"I'm sorry, I don’t have that information at the moment.\"\n\n"
        "✅ Don't repeat the user profile. Just answer the query.\n\n"
        "Do NOT repeat the user's question or profile in your reply.\n"
        "Context:\n{context}\n\n"
        "Provide a helpful, concise, and friendly recommendation."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    result = chain.invoke({"input": userQuery})
    return result['answer']
