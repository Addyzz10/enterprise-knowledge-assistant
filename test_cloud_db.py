import chromadb

client = chromadb.PersistentClient(path="db")

print("Collections:")
print(client.list_collections())