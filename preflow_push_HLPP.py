import queue
from utils import build_residual_network, get_cuts

def preflow_push(G, s, t):
    """HLPP 时间复杂度O(n^3)
    """
    # Initialize/reset the residual network.
    g = build_residual_network(G)  # g残余网络
    n = G.number_of_nodes()  # n 顶点数
    
    heights = {}
    extra = {node: 0 for node in g.nodes()}  # 顶点超额流量
    que = queue.PriorityQueue()  # 优先队列(默认是最小堆) 存储超额流节点
    gap = [0 for _ in range(2 * n)]  # gap优化，提升明显
    
    def reverse_bfs(src):
        """Perform a reverse breadth-first search from src in the residual network.
        """
        heights.clear()
        heights[src] = 0
        q = queue.Queue()
        q.put(src)
        while not q.empty():
            v = q.get()
            for u in g.predecessors(v):
                capacity = g.edges[u, v]["capacity"]
                if u not in heights and capacity > 0:
                    heights[u] = heights[v] + 1
                    q.put(u)
    
    def push(u, v):
        capacity = g.edges[u, v]["capacity"]
        # 如果v是除s, t之外的节点，且v之前不在队列中(extra[v] == 0)则入队
        if v != t and v != s and extra[v] == 0:
            que.put((-heights[v], v))
        flow = min(extra[u], capacity)
        extra[u] -= flow
        extra[v] += flow
        g.edges[u, v]["capacity"] -= flow
        g.edges[v, u]["capacity"] += flow
    
    def relabel(u):
        # 将node 高度设置为 满足capacity > flow 的高度最低的边终点高度 + 1，
        # 不会破坏 残余网络图中可以增加流量的边(u->v)，一定满足height[u] <= height[v] + 1这一条件
        min_height = 2 * n
        for v in g.successors(u):
            capacity = g.edges[u, v]["capacity"]
            if capacity > 0:
                min_height = min(min_height, heights[v])
        heights[u] = min_height + 1
    
    def gap_heuristic(height):
        """Apply the gap heuristic. Move all nodes at levels (height + 1) to max_height to level n + 1.
        """
        for node in g.nodes():
            if node != s and height < heights[node] <= n:
                heights[node] = n + 1
        for i in range(height + 1, n + 1):
            gap[i] = 0
    
    def discharge(u):
        """Discharge a node until it becomes inactive or, during phase 1 (see below), its height
         reaches at least n. The node is known to have the largest height among active nodes.
        """
        while extra[u] > 0:
            for v in g.successors(u):  # 当前弧优化
                capacity = g.edges[u, v]["capacity"]
                if heights[u] == heights[v] + 1 and capacity > 0:
                    push(u, v)
                    if extra[u] == 0:
                        break
            
            if extra[u] > 0:  # 仍有extra流量，relabel
                h = heights[u]
                relabel(u)
                gap[h] -= 1
                gap[heights[u]] += 1
                if h < n and gap[h] == 0:
                    gap_heuristic(h)
    
    # Initialize heights of the nodes.
    reverse_bfs(t)
    if s not in heights:
        # t is not reachable from s in the residual network. The maximum flow must be zero.
        return 0  # 不存在s到t的路径
    heights[s] = n
    for u in g.nodes():
        if u not in heights:
            heights[u] = n + 1
    
    extra[s] = float('inf')
    for v in g.successors(s):  # 和源点s相邻的边，push满流量
        push(s, v)
    
    for node in g.nodes():
        if node != s and node != t:
            gap[heights[node]] += 1
    # 算法终止条件，除s、t外所有节点超额流量extra=0
    while not que.empty():
        _, u = que.get()
        discharge(u)
    
    cuts = get_cuts(g, s)  # 获取最小割结果
    return extra[t], [cuts, set(g.nodes()) - cuts]
