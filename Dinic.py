import queue
from utils import build_residual_network, get_cuts

def dinic(G, s, t):
    """
    Dinic算法, 在分层网络图中找增广路。
    当G的分层结构不变时，每次增广会使增广路上剩余容量最小的一条边满流，时间复杂度O(n), 增广路数目-1，
    最多存在m条增广路（m条边），每次增广时间复杂度为O(nm)。
    每次增广会使增广路的长度d(f)（即网络图的层数）至少 + 1, 增广路最远距离为n，总算法时间复杂度O(n^2 m)
    """
    
    residual_network = build_residual_network(G)
    
    layers = {}  # layers记录顶点层数
    cur = {}  # cur用于当前弧优化
    
    # bfs进行分层
    def bfs():
        layers.clear()  # 清除之前的层次记录
        layers[s] = 0
        que = queue.Queue()
        que.put(s)
        while not que.empty():
            u = que.get()
            if u == t:
                break
            for v in residual_network.successors(u):
                capacity = residual_network.edges[u, v]["capacity"]
                if v not in layers and capacity > 0:
                    layers[v] = layers[u] + 1
                    que.put(v)
        # 返回true说明存在s->t的路径，可以继续增广，否则返回False
        return t in layers

    # dfs在分层网络上找增广路
    def dfs(u, max_flow_in):
        if u == t:
            return max_flow_in
        flow_out = 0  # 从node总的流出量
        successors = list(residual_network.successors(u))
        for i in range(cur[u], len(successors)):
            cur[u] = i  # 当前弧优化
            v = successors[i]
            capacity = residual_network.edges[u, v]["capacity"]
            # 只访问下一层节点，capacity > 0表示可以增加流量
            if layers.get(v, 0) == layers[u] + 1 and capacity > 0:
                f_branch = dfs(v, min(max_flow_in, capacity))
                residual_network.edges[u, v]["capacity"] -= f_branch
                residual_network.edges[v, u]["capacity"] += f_branch
                max_flow_in -= f_branch
                flow_out += f_branch
                if max_flow_in == 0:
                    break
                # 多路增广，如果max_flow_in没有用完，继续向下一分支找增广路。
        return flow_out

    total_flow = 0
    while True:
        if bfs() is False:  # bfs优化排除无解的情况
            break
        cur = {node: 0 for node in G.nodes()}
        flow = dfs(s, float('inf'))
        total_flow += flow

    cuts = get_cuts(residual_network, s)
    return total_flow, [cuts, set(G.nodes()) - cuts]
    