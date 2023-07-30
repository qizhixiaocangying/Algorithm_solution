# 线段树、树状数组和ST表

| 数据结构 |    下标     | 数组大小 |       初始化方式       |            使用条件            |
| :------: | :---------: | :------: | :--------------------: | :----------------------------: |
|  线段树  | 从 $1$ 开始 |   $4n$   | 分治，$build(s, t, p)$ |               \                |
| 树状数组 | 从 $1$ 开始 |  $n+1$   |        循环建立        |       满足结合律且可差分       |
|   ST表   | 从 $0$ 开始 |   $n$    | 先循环 $j$ 再循环 $i$  | $x \ opt\ x = x$，且满足结合律 |

### ST表

```python
for j in range(1, 20):
    for i in range(n - (1 << j - 1)):
        st[i][j] = max(st[i][j - 1], st[i + (1 << j - 1)][j - 1])
l, r = l - 1, r - 1
s = logn[r - l + 1]
p = max(st[l][s], st[r - (1 << s) + 1][s])
```



### 树状数组

```python
def lowbit(x):
    return x & -x

def build():
    for i in range(1, n + 1):
        b[i] += a[i - 1]
        j = i + lowbit(i)
        if j <= n:
            b[j] += b[i]

def read(i):
    ans = 0
    while i > 0:
        ans += b[i]
        i -= lowbit(i)
    return ans

def update(i, val):
    while i <= n:
        b[i] += val
        i += lowbit(i)
```



### 线段树

```python
d = [0] * (4 * n)
b = [0] * (4 * n)

def build(s, t, p):
    if s == t:
        d[p] = a[s - 1]
        return
    m = s + t >> 1
    build(s, m, p * 2)
    build(m + 1, t, p * 2 + 1)
    d[p] = d[p * 2] + d[p * 2 + 1]

def pushdown(s, t, m, p):
    if s != t and b[p]:
        d[p * 2] += b[p] * (s - m + 1)
        d[p * 2 + 1] += b[p] * (t - m)
        b[p * 2] += b[p]
        b[p * 2 + 1] += b[p]
        b[p] = 0

def query(l, r, s, t, p):
    if l <= s and t <= r:
        return d[p]
    ans = 0
    m = s + t >> 1
    pushdown(s, t, m, p)
    if m >= l:
        ans += query(l, r, s, m, p * 2)
    if r > m:
        ans += query(l, r, m + 1, t, p * 2 + 1)
    return ans

def update(l, r, s, t, p, v):
    if l <= s and t <= r:
        b[p] += v
        d[p] += (t - s + 1) * v
        return
    m = s + t >> 1
    pushdown(s, t, m, p)
    if l <= m:
        update(l, r, s, m, p * 2, v)
    if r > m:
        update(l, r, m + 1, t, p * 2 + 1, v)
    d[p] = d[p * 2] + d[p * 2 + 1]


build(1, n, 1)
```



### 离散化

数据范围大，但是数组长度不大，排序，然后将数组中的数字利用二分或哈希表映射为下标。

### 逆序对

维护一个数组 $b$，表示 $i$ 右边的每个数 $x$ 出现的次数，则$j>i \ 且\ nums[i]>nums[j]$ 的逆序对 $(i,j)$ 个数为 $sum(d[1:nums[i] - 1])$，通过树状数组/线段树求此前缀和，当 $i$ 移动时，$d[nums[i]]++$，同样通过树状数组/线段树实现。

### 线段树：动态开点

类似二叉树。

```python
class Node:
    def __init__(self):
        self.val = 0
        self.left = None
        self.right = None
        self.lazy = 0
```

### 树状数组区间加区间和

<img src="%E7%BA%BF%E6%AE%B5%E6%A0%91%E3%80%81%E6%A0%91%E7%8A%B6%E6%95%B0%E7%BB%84%E5%92%8CST%E8%A1%A8.assets/image-20230319205907623.png" alt="image-20230319205907623" style="zoom:50%;" />

### 易错点

线段树的 $update$ 函数最后不要忘记更新 $d[p]$。

$l,r,s,t$ 不要写错了。

线段树初始化时是 ```tree[p] = a[s - 1]``` 不要写成 ```p - 1``` 了。 

是否递归左右子树是看 $l,r$ 和 $m$ 的关系。

### 1.力扣：[327. 区间和的个数 ](https://leetcode.cn/problems/count-of-range-sum/description/)

### 题目：

<img src="%E7%BA%BF%E6%AE%B5%E6%A0%91%E3%80%81%E6%A0%91%E7%8A%B6%E6%95%B0%E7%BB%84%E5%92%8CST%E8%A1%A8.assets/image-20230318190201295.png" alt="image-20230318190201295" style="zoom:67%;" />

### 题解一：树状数组

算法思路：

设 $nums$ 的前缀和数组为 $pre$，则我们要求的是满足 $lower \leqslant pre[r]-pre[l-1] \leqslant upper$ 的区间 $[l,r]$ 的个数。变换得：$pre[r]-upper \leqslant pre[l-1] \leqslant pre[r]-lower$。

容易想到维护一个树状数组，表示每个元素出现的次数，然后枚举右区间端点 $r$ ，利用树状数组求出区间 $[pre[r]-upper,pre[r]-lower]$ 的和，这个和即为原数组中所有以 $r$ 为右端点且满足条件的区间和的个数。

这个题目有几点需要注意：

1. $pre$ 的取值范围较大，所以我们应该要做离散化
2. 离散化需要对 $pre[i]-upper,pre[i],pre[i]-lower$ 都进行
3. 由于 $l$ 不可能出现在 $r$ 的右边，因此 $l-1$ 最大为 $r-1$，所以我们在枚举 $r$ 时需要先 $add(pre[r-1])$，然后再 $query([pre[r]-upper,pre[r]-lower])$ 。

代码实现：

```python
class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        def lowbit(x):
            return x & - x
        
        def add(x):
            while x <= n:
                tree[x] += 1
                x += lowbit(x)
        
        def query(x):
            ans = 0
            while x > 0:
                ans += tree[x]
                x -= lowbit(x)
            return ans

        pre = [0] + list(itertools.accumulate(nums))
        ranges = []
        for i in range(1, len(pre)):
            ranges.append(pre[i - 1])
            ranges.append(pre[i] - upper)
            ranges.append(pre[i] - lower)
        ranges = sorted(list(set(ranges)))
        n, res = len(ranges), 0
        dic = {}
        for i in range(n):
            if ranges[i] not in dic:
                dic[ranges[i]] = i + 1
        tree = [0] * (n + 1)
        for i in range(1, len(pre)):
            p, q = pre[i], pre[i - 1]
            add(dic[q])
            res += (query(dic[p - lower]) - 
                    query(dic[p - upper] - 1))
        return res
```



### 题解二：有序集合+正难则反

算法思路：

从右到左遍历 $nums$ ，用一个有序集合保存以当前元素开头的所有区间和，由于集合有序，我们可以利用二分法求出大小在 $[lower,upper]$ 之间的区间和的数量。

当遍历到某一个元素时，集合所需要做的操作是将集合中所有元素加上当前元素 $nums[i]$，然后添加 $nums[i]$ 到集合中。由于将集合中所有元素加上 $nums[i]$ 时间复杂度太高，我们不妨换个思路，将区间范围向左偏移 $nums[i]$ 个单位，即为 $[lower-nums[i], upper-nums[i]]$，这与给每个元素加上 $nums[i]$ 是等价的，即加上 $nums[i]$ 后在区间 $[lower, upper]$ 中的元素个数等于原始元素在区间 $[lower-nums[i], upper-nums[i]]$ 中的个数。由于 $nums[i]$ 是新添加的元素，因此 $nums[i]$ 也需要偏移。

在编码时，我们用一个变量 $d$ 表示当前偏移量。

代码实现：

```python
from sortedcontainers import SortedList

class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        ans = d = 0
        ls = SortedList()
        for c in nums[::-1]:
            d += c
            ls.add(c - d)
            r = ls.bisect_right(upper - d)
            l = ls.bisect_left(lower - d)
            ans += r - l
        return ans
```



### 2.力扣：[406. 根据身高重建队列](https://leetcode.cn/problems/queue-reconstruction-by-height/)

### 题目：

<img src="%E7%BA%BF%E6%AE%B5%E6%A0%91%E3%80%81%E6%A0%91%E7%8A%B6%E6%95%B0%E7%BB%84%E5%92%8CST%E8%A1%A8.assets/image-20230318202631486.png" alt="image-20230318202631486" style="zoom:60%;" />

### 题解一：贪心法

算法思路：

矮的不管怎么放都不会对高的产生影响，同样高度的互相会产生影响，因此我们按高度从高到低再按 $k$ 从小到大排序。遍历排序后的数组根据 $k$ 往答案数组里插入即可。

代码实现：

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x: (-x[0], x[1]))
        ans = []
        for ls in people:
            ans.insert(ls[1], ls)
        return ans
```



### 题解二：树状数组+二分答案

算法思路：

我们设置一个数组 $ans$ ，这个数组表示最终每个人放的位置，初始时每个元素为空。我们按高度从低到高，$k$ 从大到小进行排序，我们可以先放矮的，在 $ans$ 中预留 $k$ 个空位置给后续高的放，我们如果每次都遍历 $k$ 个空位置的话，那么时间复杂度就会达到 $O(n^2)$ 。有没有更好的办法呢？我们可以维护一个树状数组，表示哪些位置已经被放，易知位置越大，前面的空位置数更大，满足单调性，因此我们可以利用二分法求出空位置刚好为 $k$ 的放置位置，然后将该位置设置为非空即可。

代码实现：

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        def lowbit(x):
            return x & -x

        def query(x):
            ans = 0
            while x > 0:
                ans += tree[x]
                x -= lowbit(x)
            return ans
        
        def add(x):
            while x <= n:
                tree[x] -= 1
                x += lowbit(x)
            
        n = len(people)
        tree = [0] * (n + 1)
        for i in range(1, n + 1):
            tree[i] += 1
            j = i + lowbit(i)
            if j <= n:
                tree[j] += tree[i]
        ans = [[] for _ in range(n)]
        people.sort(key=lambda x: (x[0], -x[1]))
        for ls in people:
            l, r = 1, n + 1
            k = ls[1] + 1
            while l < r:
                mid = l + r >> 1
                if query(mid) < k: l = mid + 1
                else: r = mid
            ans[l - 1] = ls
            add(l)
        return ans
```



### 3.洛谷：[P8773 [蓝桥杯 2022 省 A] 选数异或](https://www.luogu.com.cn/problem/P8773)

### 题目：

![image-20230319103957180](%E7%BA%BF%E6%AE%B5%E6%A0%91%E3%80%81%E6%A0%91%E7%8A%B6%E6%95%B0%E7%BB%84%E5%92%8CST%E8%A1%A8.assets/image-20230319103957180.png)

### 题解一：ST表

算法思路：

设 $a_i \oplus t=x$，则 $t = x\oplus a_i$，要求 $[l,r]$ 中是否有两个数的异或等于 $x$，只需求 $[l,r]$ 中是否有 $a_j=t\ (t=x\oplus a_i，i\in [l,r])$。要求 $t$ 是否在 $nums[l:r+1]$ 中，只需要求 $t$ 的最大下标 $j$ 是否在 $[l, r]$ 中。设一个数组为 $pos$， $pos_i$ 表示 $[1,i]$ 中 $x\oplus a_i，i\in [l,r]$ 的最大下标，对于每次询问，我们只需要求 $pos[l:r+1]$ 中的最大值是否 大于等于 $l$。实际求解中可以使用 $ST$ 表，$pos$ 数组可以省略。

代码实现：

```python
import sys

n, m, x = map(int, input().split())
a = list(map(int, input().split()))
st = [[-1] * 20 for _ in range(n)]
logn = [0] * (n + 2)
logn[1] = 0
dic = {}
for i in range(n):
    dic[a[i]] = i
    c = a[i] ^ x
    st[i][0] = dic.get(c, -1)
    logn[i + 2] = logn[i // 2 + 1] + 1
for j in range(1, 20):
    for i in range(n - (1 << j - 1)):
        st[i][j] = max(st[i][j - 1], st[i + (1 << j - 1)][j - 1])
for _ in range(m):
    l, r = map(int, sys.stdin.readline().strip().split())
    l, r = l - 1, r - 1
    s = logn[r - l + 1]
    p = max(st[l][s], st[r - (1 << s) + 1][s])
    print('yes' if p >= l else 'no')

```



### 题解二：动态规划

算法思路：

同题解一，设 $dp[i]$ 为 $[1,i]$ 中 $pos$ 的最大值，每次询问时只需要判断 $dp[r]$ 是否大于等于 $l$ 即可，因为 $pos[i] \leqslant i$，所以若 $dp[r] \geqslant l$，即 $[1,r]$ 中存在 $pos[j] \geqslant l$，则必有 $j \geqslant l$，即 $j \in [l,r]$。

代码实现： 

```python
import sys

n, m, x = map(int, input().split())
a = list(map(int, input().split()))
dp = [0] * (n + 1)
dic = {}
for i in range(n):
    dic[a[i]] = i + 1
    c = a[i] ^ x
    dp[i + 1] = max(dp[i], dic.get(c, 0))
for _ in range(m):
    l, r = map(int, sys.stdin.readline().strip().split())
    print('yes' if dp[r] >= l else 'no')

```



