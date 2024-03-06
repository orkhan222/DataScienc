import redis
import networkx as nx
import matplotlib.pyplot as plt

redis_client = redis.StrictRedis(host='91.102.161.166', port=6379, db=0)

redis_client.set('OrkhanMustafayev', '09-11-2001')

G = nx.Graph()

G.add_node("Orkhan")
G.add_node("Ali")
G.add_node("Farid")
G.add_node("Nigar")

G.add_edge("Orkhan", "Farid")
G.add_edge("Farid", "Ali")
G.add_edge("Nigar", "Ali")
G.add_edge("Farid", "Nigar")
G.add_edge("Orkhan", "Nigar")
G.add_edge("Orkhan", "Ali")

nx.draw(G, with_labels=True)
plt.show()