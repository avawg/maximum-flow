import time
import os
import networkx as nx

from preflow_push_FIFO import preflow_push
# from preflow_push_HLPP import preflow_push
from utils import read_edgelist, measure
from triangle_util import get_triangles, get_triangle_node_weight

def exact(G, k=3):
    """
    Maximum Weighted Triangle Density的精确算法，时间复杂度为O(nmlog(W))，
    :param G: 输入的无向图
    :param k: clique的大小，k=3, clique为triangle, 即为MWTD
                           k=2, clique为edge, 对应MWED
    :return: subG: subgraph with maximum Weighted triangle density
    """
    if k not in [2, 3]:
        raise Exception("参数k只能为2或3")
    
    def construct_digraph(clique_weight, lambda_):
        """
        构建有向图，只调用一次
        时间复杂度
        :param clique_weight: {}，key为clique元组， value为对应的权重
        :param lambda_:
        :return:
            g: 构建的有向带权图
        """
        g = nx.DiGraph()
        # 初始化有向图节点
        g.add_node('s')
        for clique in clique_weight:
            g.add_node(clique)
        for node in G.nodes():
            g.add_node(node)
        g.add_node('t')
        # 添加边
        for clique, weight in clique_weight.items():
            g.add_edge('s', clique, capacity=weight)
            for node in clique:
                g.add_edge(clique, node, capacity=float('inf'))
        for node in G.nodes():
            g.add_edge(node, 't', capacity=lambda_)
        return g
    
    def update_digraph(g, lambda_):
        for node in g.pred['t']:  # 存在该边则直接更新
            g.add_edge(node, 't', capacity=lambda_)
    
    # the set of cliques
    clique_weight = {}
    if k == 3:
        triangles = get_triangles(G)
        clique_weight, _ = get_triangle_node_weight(G, triangles)
    elif k == 2:
        for edge in G.edges():
            public_nodes = set(G.adj[edge[0]]) & set(G.adj[edge[1]])
            clique_weight[tuple(edge)] = len(public_nodes)
    
    # cliques为空集，直接返回
    if len(clique_weight) == 0:
        return None
    
    lo = 0
    hi = sum(clique_weight.values())
    lambda_ = (lo + hi) / 2
    n = G.number_of_nodes()
    condition = 1 / (n * (n - 1))
    
    g = construct_digraph(clique_weight, lambda_)
    V = {}
    # 二分查找
    while hi - lo >= condition:
        # cut_value, minimum_cuts = nx.minimum_cut(g, 's', 't')
        cut_value, minimum_cuts = preflow_push(g, 's', 't')
        
        # S 为源点's'所在集合
        S = minimum_cuts[0] if 's' in minimum_cuts[0] else minimum_cuts[1]
        
        if S == {'s'}:  # 所有子图的密度都比lambda_小
            hi = lambda_
        else:
            lo = lambda_  # 存在子图的密度 >= lambda_
            V = set(G.nodes()) & S
        lambda_ = (lo + hi) / 2
        
        update_digraph(g, lambda_)
    return G.subgraph(V)


if __name__ == '__main__':
    path = "./data/"
    for file in os.listdir(path):
        datasets = file.split('.')[0]
        
        file_name = path + file
        G = read_edgelist(file_name)
        
        start = time.perf_counter()
        subG = exact(G, k=2)
        end = time.perf_counter()
        
        mes = {"datasets_name": datasets}
        mes.update(measure(subG))
        mes.update({"run time(ms)": int((end - start) * 1000 * 100) / 100})
        print(mes)
    