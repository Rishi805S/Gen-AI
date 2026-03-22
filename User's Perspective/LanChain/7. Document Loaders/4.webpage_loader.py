# --------- Only used for Static web pages ------------
# from langchain_community.document_loaders import WebBaseLoader
# import os
# os.environ["USER_AGENT"] = "my-app"
# url ='https://en.wikipedia.org/wiki/Static_web_page'
# loader = WebBaseLoader(url)
# docs = loader.load()
# print(docs[0].page_content)

# -------- This worked but browser is opened -----------
# from langchain_community.document_loaders import PlaywrightURLLoader
# loader = PlaywrightURLLoader(
#     urls=["https://medium.com/@shivambhadani_/system-design-for-beginners-everything-you-need-in-one-article-c74eb702540b"],
#     headless=False
# )
# docs = loader.load()
# print(docs[0].page_content[:500])

