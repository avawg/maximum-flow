import queue
import networkx
from triangle_util import get_triangles

# 读图(无向图)
def read_edgelist(file_name, separator='\t'):
    """
    从文件中读取创建图
    :param file_name: 文件名
    :param separator: 分隔符
    :return: G: 创建的无向图
    """
    G = networkx.Graph()
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if line == "" or line[0] == "#":  # 过滤掉空行和注释行
                continue
            cols = line.split(separator)
            s, t = cols[0], cols[1]
            weight = int(cols[2]) if len(cols) > 2 else 1
            if s == t:  # 过滤掉自循环边
                continue
            # 添加节点和边
            G.add_node(s)
            G.add_node(t)
            G.add_edge(s, t, weight=weight)
    
    return G

# 构建初始的残余网络
def build_residual_network(G):
    residual_network = networkx.DiGraph()
    for edge in G.edges():
        u, v = edge
        residual_network.add_edge(u, v, capacity=G.edges[u, v]["capacity"])
        residual_network.add_edge(v, u, capacity=0)
    return residual_network

# 根据最大流时残余网络获取最小割结果
def get_cuts(residual_network, s):
    cuts = set()  # 源点's'所在集合
    cuts.add(s)
    que = queue.Queue()
    que.put(s)
    
    while not que.empty():
        u = que.get()
        for v in residual_network.successors(u):
            if v not in cuts and residual_network.edges[u, v]["capacity"] > 0:
                cuts.add(v)
                que.put(v)
    return cuts

# 子图稠密指标
def measure(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    t = len(get_triangles(G))
    # 边密度
    edge_density = m / (n * (n - 1) / 2) if n > 1 else 0
    # 三角形密度
    triangle_density = t / (n * (n - 1) * (n - 2) / 6) if n > 2 else 0
    
    return {
        "edge_density": int(edge_density * 100) / 100,
        "triangle_density": int(triangle_density * 100) / 100,
        "number_of_nodes": n
    }
    