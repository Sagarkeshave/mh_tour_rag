from langchain_google_genai import GoogleGenerativeAI

api_key = "AIzaSyCP4WNid41ugJqxP_TcyMGeSQ-4VWRWglY"
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
print(
    llm.invoke(
        "What are some of the pros and cons of Python as a programming language in just 6 bullet points"
    )
)
