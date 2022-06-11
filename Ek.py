from utils import build_residual_network, get_cuts
import queue

def edmonds_karp(G, s, t):
    """
    (1) f是图G的最大流
    (2) 存在(s -> t)的flow，其值等于 (s, t)-cut (A, B)的容量
        割: 将图G的顶点分成两个集合(A, B)，其容量为 sum(capacity_edge_uv)
        即所有割边u->v的容量和，u属于A, v属于B
    (3) 残余网络图Gf中不存在增广路

    这三条结论是等价的。
    证明：
        (2) -> (1): f = flow:A->B - flow:B->A 而flow:A->B <= (s-t)-cut (A, B)的容量，flow:B->A >= 0
        (1) -> (3): 反证法，假设Gf中存在增广路，说明f可以继续增大
        (3) -> (2): 当Gf中不存在增广路时，令A = {s}，从s开始bfs搜索扩展A，只要A中顶点到u的边flow < capacity,
        则将u加入A中。最终B = V - A, 可知t属于B(不存在s->t的增广路)。
        这时所有边u->v(u属于A，v属于B) flow=capacity, 所有边v->u(u属于A，v属于B) flow=0，否则会增加反向边，从而v可以拓展加入A。
        所以此时f = capacity((s-t)-cut (A, B))

    故最大流最小割问题是等价的，最小割问题先求最大流再在残余网络Gf中应用一次bfs。
    而求最大流问题，则可转化为在Gf中不断找增广路，直到无增广路，则求得最大流。

    Ek算法通过bfs找增广路，每次找到的增广路是距离最短的，直到在残余网络中不存在增广路。
    至多m次增广会使增广路的长度d(f) + 1，因为每次增广会使增广路剩余容量最小的一条边满流，
    增广路最远距离为n，每次增广时间复杂度为O(m) 故总时间复杂度为O(nm^2)
    :param G: 有向图G
    :param s:
    :param t:
    :return: total_fow 最大流大小
    """
    
    residual_network = build_residual_network(G)
    
    pre = {}  # 记录流传入边
    flows = {}  # 记录流大小
    
    def bfs():
        que = queue.Queue()
        que.put(s)
        flows[s] = float('inf')
        pre[s] = -1
        while not que.empty():
            u = que.get()
            if u == t:  # 可以从s找到t节点
                return True
            for v in residual_network.successors(u):
                capacity = residual_network.edges[u, v]["capacity"]
                # v not in pre表示顶点v未被访问过，capacity > 0表示可以增加流量
                if v not in pre and capacity > 0:
                    pre[v] = u
                    flows[v] = min(flows[u], capacity)
                    que.put(v)
        return False

    total_flow = 0
    while True:
        pre.clear(), flows.clear()
        if bfs() is False:
            break
        flow = flows[t]
        total_flow += flow
        # 更新残余网络
        v = t
        while v != s:
            u = pre[v]
            residual_network.edges[u, v]["capacity"] -= flow  # 正向边加上增广路径流量
            residual_network.edges[v, u]["capacity"] += flow  # 反向边减去增广路径流量
            v = u
    cuts = get_cuts(residual_network, s)
    return total_flow, [cuts, set(G.nodes()) - cuts]
