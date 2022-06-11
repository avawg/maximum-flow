def get_triangles(G):
    """
    list all triangle of undirected graph G using forward algorithm
    :return: triangle: list type, all triangles in graph,
                [ (vertex1, vertex2, vertex3), ... ]
    """
    triangles = []
    adjacency = G.adj
    # sort vertex in decreasing order of degree
    d = sorted(adjacency.items(), key=lambda item: len(item[1]), reverse=True)
    ordered_vertex = [t[0] for t in d]
    del d
    # initialize A with vertex
    A = {vertex: set() for vertex in ordered_vertex}
    visited = set()
    for v in ordered_vertex:
        for u in adjacency[v]:
            if u in visited:
                continue
            # print(v, A[v], u, A[u])
            t = A[u] & A[v]
            for w in t:
                triangle = tuple(sorted((v, u, w)))
                triangles.append(triangle)
            A[u].add(v)
        visited.add(v)
    return triangles

def get_triangle_node_weight(G, triangles):
    """
    求每个triangle和节点的权重
    triangle 权重为其参与的4-clique个数
    node 权重为其参与triangle权重之和
    """
    edge_adj_nodes = get_edge_adj_nodes(G, triangles)
    
    triangle_weight = {}
    node_weight = {node: 0 for node in G.nodes()}
    for triangle in triangles:
        u, v, w = triangle
        node_set1 = edge_adj_nodes[(u, v)]
        node_set2 = edge_adj_nodes[(u, w)]
        node_set3 = edge_adj_nodes[(v, w)]
        
        # 公共顶点集大小，即为triangle参与4-clique的个数
        public_nodes = node_set1 & node_set2 & node_set3
        triangle_weight[triangle] = len(public_nodes)
        
        # public_sets中的点，与triangle组成4clique
        for node in public_nodes:
            node_weight[node] += triangle_weight[triangle]
    
    return triangle_weight, node_weight

def get_node_adj_edges(G, triangles):
    """
    计算与顶点构成triangle的边的集合
    """
    node_adj_edges = {node: set() for node in G.nodes()}
    for triangle in triangles:
        u, v, w = triangle
        node_adj_edges[u].add((v, w))
        node_adj_edges[v].add((u, w))
        node_adj_edges[w].add((u, v))
    return node_adj_edges

def get_edge_adj_nodes(G, triangles):
    """
    计算与边构成triangle的顶点的集合
    """
    edge_adj_nodes = {tuple(sorted(edge)): set() for edge in G.edges()}
    for triangle in triangles:
        u, v, w = triangle
        edge_adj_nodes[(u, v)].add(w)
        edge_adj_nodes[(u, w)].add(v)
        edge_adj_nodes[(v, w)].add(u)
    return edge_adj_nodes
    