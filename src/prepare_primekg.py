import pandas as pd

print("--- Step 1: Cleaning Nodes ---")
nodes = pd.read_csv('nodes.csv')
# Rename columns for Neo4j: node_index becomes the ID, group becomes the LABEL
nodes.columns = ['node_index:ID', 'node_id', ':LABEL', 'node_name', 'node_source']
nodes.to_csv('nodes_cleaned.csv', index=False)

print("--- Step 2: Removing Duplicate Edges ---")
edges = pd.read_csv('edges.csv')
# Standardize column names for Neo4j
edges.columns = ['relation', 'display_relation', ':START_ID', ':END_ID']
# Create a 'set' of start/end/relation to find duplicates regardless of direction
# (This follows the article's advice for an undirected graph)
edge_group = edges.apply(lambda x: frozenset([x[':START_ID'], x[':END_ID'], x['relation']]), axis=1)
edges_cleaned = edges.groupby(edge_group).first()

# Rename 'relation' to ':TYPE' for Neo4j
edges_cleaned = edges_cleaned.rename(columns={'relation': ':TYPE'})
edges_cleaned.to_csv('edges_cleaned.csv', index=False)

print("Done! You now have 'nodes_cleaned.csv' and 'edges_cleaned.csv'.")