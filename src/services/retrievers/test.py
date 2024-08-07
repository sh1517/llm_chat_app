import os
from retriever_manager import create_retrives
from langchain_community.document_loaders import TextLoader

# loader = TextLoader("databases/toyota_manual.txt", encoding='utf-8')
# toyota_manual_content = loader.load()
# # print(toyota_manual_content)

# toyota_retrieval = create_retrives(toyota_manual_content)
# # print(toyota_retrieval)
# res = toyota_retrieval.invoke("error code 51")
# print(res)

# file_path = os.path.join(os.getcwd(), "databases/toyota_manual.txt")
# if not os.path.exists(file_path):
#     print("File does not exist: ", file_path)
# else:
#     print("File exist: ", file_path)
#     loader = TextLoader(file_path, encoding='utf-8')
#     toyota_manual_content = loader.load()
#     print(toyota_manual_content)


base_dir = os.path.abspath(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
manual_path = os.path.join(project_dir, "databases", "toyota_manual.txt")
# print(manual_path)

loader = TextLoader(manual_path, encoding='utf-8')
toyota_manual_content = loader.load()
# print(toyota_manual_content)

toyota_retrieval = create_retrives(toyota_manual_content)
res = toyota_retrieval.invoke("error code 51")
print(res)