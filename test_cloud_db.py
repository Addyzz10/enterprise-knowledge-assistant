import chromadb

client = chromadb.PersistentClient(path="./db")

print(client.list_collections())