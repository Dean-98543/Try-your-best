## **剑指offer66总结**

#### 03. 数组中重复的数字

找出数组中重复的数字。

在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

**示例 1：**

```
输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3 
```

 **限制：**

```
2 <= n <= 100000
```

##### 解法一：原地交换

```python
# Python3
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        N = len(nums)
        for i in range(N):
            while nums[i] != i:         # 发现这个坑里的萝卜不是自己家的
                temp = nums[i]          # 看看你是哪家的萝卜
                if nums[temp] == temp:  # 看看你家里有没有和你一样的萝卜
                    return temp         # 发现你家里有了和你一样的萝卜，那你就多余了，上交国家
                else:                   # 你家里那个萝卜和你不一样
                    nums[temp], nums[i] = nums[i], nums[temp]   # 把你送回你家去，然后把你家里的那个萝卜拿回来
        return -1
```

```c++
// C++
#include <iostream>
#include <vector>
using namespace std;
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        int N = nums.size();
        for(int i=0; i<N; i++){
            while(nums[i] != i){              //发现这个坑里的萝卜不是自己家的
                int temp = nums[i];           //看看你是哪家的萝卜
                if(nums[temp] == temp)        //看看你家里有没有和你一样的萝卜
                    return temp;            //发现你家里有了和你一样的萝卜，那你就多余了，上交国家
                else                        //你家里那个萝卜和你不一样    
                    swap(nums[temp], nums[i]);  //把你送回你家去，然后把你家里的那个萝卜拿回来
            }
        }
        return -1;
    }
};
```

#### 04. 二维数组中的查找

在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

 **示例:**

现有矩阵 matrix 如下：

```
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```

给定 target = `5`，返回 `true`。

给定 target = `20`，返回 `false`。

 **限制：**

```
0 <= n <= 1000
0 <= m <= 1000
```

##### 解法一：线性查找

```python
# Python3
from typing import List
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        N, M = len(matrix), (len(matrix[0]) if matrix else 0)
        x, y = N-1, 0
        while 0<=x and y<M:
            if matrix[x][y] == target:
                return True
            elif matrix[x][y] > target:
                x-=1
            else:
                y+=1
        return False
```

```c++
// C++
#include <iostream>
#include <vector>
using namespace std;
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        int N=matrix.size(), M=!matrix.empty()?matrix[0].size():0;
        int x=N-1, y=0;
        while( 0<=x && y<M){
            if(matrix[x][y]==target)    return true;        
            else if(matrix[x][y]>target)    x-=1;
            else    y+=1;
        } 
        return false;
    }
};
```

#### 05. 替换空格

请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

 **示例 1：**

```
输入：s = "We are happy."
输出："We%20are%20happy."
```

 **限制：**

```
0 <= s 的长度 <= 10000
```

##### 解法一：遍历添加

```python
# Python3
class Solution:
    def replaceSpace(self, s: str) -> str:
        res = ""
        for c in s:
            if c == " ":
                res+="%20"
            else:
                res+=c
        return res
```

```c++
// C++
#include <iostream>
using namespace std;
class Solution {
public:
    string replaceSpace(string s) {
        string res;
        for(auto c:s){
            if(c==' ')
                res+="%20";
            else
                res+=c;
        }
        return res;
    }
};
```

#### 06. 从尾到头打印链表（简单）

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

**示例 1：**

```
输入：head = [1,3,2]
输出：[2,3,1]
```

**限制：**

```
0 <= 链表长度 <= 10000
```

##### 解法一：递归

```python
# Python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        if not head:
            return []
        else:
            return self.reversePrint(head.next) + [head.val]
```

```c++
// C++
#include <iostream>
#include <vector>
using namespace std;
//  * Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
class Solution {
public:
    vector<int> res;
    vector<int> reversePrint(ListNode* head) {
        recur(head);
        return res;
    }
    void recur(ListNode* head){
        if(head == nullptr)     return ;
        recur(head->next);
        res.push_back(head->val);
    }
};
```



##### 解法二：辅助栈/顺序/迭代

```python
# Python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        res = []
        while head:
            res.append(head.val)
            head = head.next
            # 可以使用res.reverse()，如此则return res
            return res[::-1]
```

```C++
// C++
#include <iostream>
#include <vector>
#include <stack>
using namespace std;
//  * Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
class Solution {
public:
    vector<int> reversePrint(ListNode* head) {
        stack<int> st;
        vector<int> res;
        while(head){
            st.push(head->val);
            head = head->next;
        }
        while(!st.empty()){
            res.push_back(st.top());
            st.pop();
        }
        return res;
    }
};
```

#### 07. 重建二叉树

输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

例如，给出

```
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
```

返回如下的二叉树：

```
    3
   / \
  9  20
    /  \
   15   7
```

**限制：**

```
0 <= 节点个数 <= 5000
```

##### 解法一：递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if inorder:
            root_val = preorder[0]
            root_idx = inorder.index(root_val)
            
            root = TreeNode(root_val)
            root.left = self.buildTree(preorder[1: root_idx+1], inorder[:root_idx])
            root.right= self.buildTree(preorder[root_idx+1:], inorder[root_idx+1:])
            return root
```

#### 09. 用栈实现队列

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列的支持的所有操作（`push`、`pop`、`peek`、`empty`）：

实现 `MyQueue` 类：

- `void push(int x)` 将元素 x 推到队列的末尾
- `int pop()` 从队列的开头移除并返回元素
- `int peek()` 返回队列开头的元素
- `boolean empty()` 如果队列为空，返回 `true` ；否则，返回 `false`

 **说明：**

- 你只能使用标准的栈操作 —— 也就是只有 `push to top`, `peek/pop from top`, `size`, 和 `is empty` 操作是合法的。
- 你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。

 **进阶：**

- 你能否实现每个操作均摊时间复杂度为 `O(1)` 的队列？换句话说，执行 `n` 个操作的总时间复杂度为 `O(n)` ，即使其中一个操作可能花费较长时间。

 **示例：**

```
输入：
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
输出：
[null, null, null, 1, 1, false]

解释：
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false
```

**提示：**

- `1 <= x <= 9`
- 最多调用 `100` 次 `push`、`pop`、`peek` 和 `empty`
- 假设所有操作都是有效的 （例如，一个空的队列不会调用 `pop` 或者 `peek` 操作）

##### 解法一：输入栈+输出栈

```python
# Python3
class MyQueue:
    def __init__(self):
        self.stackA = list()
        self.stackB = list()

    def push(self, x: int) -> None:
        self.stackA.append(x)

    def pop(self) -> int:
        if not self.stackB:
            while self.stackA:
                self.stackB.append(self.stackA.pop())
        return self.stackB.pop()
    def peek(self) -> int:
        if not self.stackB:
            while self.stackA:
                self.stackB.append(self.stackA.pop())
        return self.stackB[-1]
    def empty(self) -> bool:
        return not self.stackB and not self.stackA
```

```c++
// C++
class MyQueue {
private:
    stack<int> stackA, stackB;
    void in2out(){
        while(!stackA.empty()){
            stackB.push(stackA.top());
            stackA.pop();
        }
    }

public:
    MyQueue() {}

    void push(int x) {
        stackA.push(x);
    }
    
    int pop() {
        if(stackB.empty()){
            in2out();
        }
        int x=stackB.top();
        stackB.pop();
        return x;
    }
    
    int peek() {
        if(stackB.empty()){
            in2out();
        }
        return stackB.top();
    }
    
    bool empty() {
        return stackA.empty() && stackB.empty();
    }
};
```

#### 10. I斐波那契数列

写一个函数，输入 `n` ，求斐波那契（Fibonacci）数列的第 `n` 项（即 `F(N)`）。斐波那契数列的定义如下：

```
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
```

斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

 **示例 1：**

```
输入：n = 2
输出：1
```

**示例 2：**

```
输入：n = 5
输出：5
```

 **提示：**

- `0 <= n <= 100`

##### 解法一：动态规划

```python
# Python3
class Solution:
    def fib(self, n: int) -> int:
        if n<1:
            return 0
        dp = [0]*(n+1)
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n+1):
            dp[i] = (dp[i-1] + dp[i-2]) % 1000_000_007
        return dp[n]
```

```c++
// C++
#include <iostream>
using namespace std;
class Solution {
public:
    int fib(int n) {
        if(n<1) return 0;
        int dp[n+1];
        dp[0] = 0;
        dp[1] = 1;
        for(int i=2; i<n+1; i++){
            dp[i] = (dp[i-1] + dp[i-2]) % 1000000007;
        }
        return dp[n];
    }
};
```

#### 10. II青蛙跳台阶问题

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 `n` 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

**示例 1：**

```
输入：n = 2
输出：2
```

**示例 2：**

```
输入：n = 7
输出：21
```

**示例 3：**

```
输入：n = 0
输出：1
```

**提示：**

- `0 <= n <= 100`

##### 解法一：动态规划

```python
# Python3
class Solution:
    def numWays(self, n: int) -> int:
        if n<=1:    return 1
        dp = [0]*(n+1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n+1):
            dp[i] = (dp[i-1] + dp[i-2]) % 1000_000_007
        return dp[-1]
```

```c++
// C++
#include <iostream>
using namespace std;
class Solution {
public:
    int numWays(int n) {
        if(n<=1) return 1;
        int dp[n+1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i=2; i<n+1; i++){
            dp[i] = (dp[i-1] + dp[i-2]) % 1000000007;
        }
        return dp[n];
    }
};
```

#### 18. 删除链表的节点（简单）

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点。

**注意：**此题对比原题有改动

**示例 1:**

```
输入: head = [4,5,1,9], val = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
```

**示例 2:**

```
输入: head = [4,5,1,9], val = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
```

**说明：**

- 题目保证链表中节点的值互不相同
- 若使用 C 或 C++ 语言，你不需要 `free` 或 `delete` 被删除的节点

##### 解法一：单指针遍历

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if head.val==val:
            return head.next
        else:
            p = head
            while p.next:
                if p.next.val == val:
                    p.next = p.next.next
                    return head
                else:
                    p = p.next
```

##### 解法二：双指针遍历

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if head.val==val:
            return head.next
        else:
            pre, cur = head, head.next
            while cur and cur.val!=val:
                pre, cur = cur, cur.next
            if cur.val==val:
                pre.next = cur.next
            return head
```

#### 22. 链表中倒数第k各节点（简单）

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。

**示例：**

```
给定一个链表: 1->2->3->4->5, 和 k = 2.

返回链表 4->5.
```

##### 解法一：双指针

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return None
        left = right = head
        for _ in range(k):
            right = right.next
        while right:
            left, right = left.next, right.next
        return left
```

#### 24. 反转链表（简单）

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

**示例:**

```
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```

**限制：**

```
0 <= 节点个数 <= 5000
```

##### 解法一：迭代

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head:
            return None
        elif not head.next:
            return head
        else:
            pre, cur = None, head
            while cur:
                temp = cur.next
                cur.next = pre
                pre, cur = cur, temp
            return pre
```

##### 解法二：递归

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head:
            return None
        elif not head.next:
            return head
        node = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return node
```

#### 25. 合并两个有序链表（简单）

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

**示例1：**

```
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```

**限制：**

```
0 <= 链表长度 <= 1000
```

##### 解法一：迭代

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        elif not l2:
            return l1
        else:
            prehead = head = ListNode(0)
            while l1 and l2:
                if l1.val < l2.val:
                    head.next = l1
                    l1 = l1.next
                else:
                    head.next = l2
                    l2 = l2.next
                head = head.next
            head.next = l1 if l1 else l2
            return prehead.next
```

##### 解法二：递归

```python  
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        elif not l2:
            return l1
        elif l1.val<l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```

#### 32-I. 从上到下打印二叉树（中等）：直接打印出所有节点

从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

例如:
给定二叉树: `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回：

```
[3,9,20,15,7]
```

**提示：**

1. `节点总数 <= 1000`

##### 解法一：层次遍历

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        res = []
        from collections import deque	# 使用双端队列
        queue = deque()
        queue.append(root)
        while queue:
            for _ in range(len(queue)):
                node =  queue.popleft()
                res.append(node.val)
                if node.left:   queue.append(node.left)
                if node.right:  queue.append(node.right)
        return res
```

#### 32-II. 从上到下打印二叉树（简单）：按层顺序打印节点

从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

例如:
给定二叉树: `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回其层次遍历结果：

```
[
  [3],
  [9,20],
  [15,7]
]
```

**提示：**

1. `节点总数 <= 1000`

##### 解法一：层次遍历

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        res = []
        from collections import deque	# 使用双端队列
        queue = deque([root])
        while queue:
            temp = []
            for _ in range(len(queue)):
                node =  queue.popleft()
                temp.append(node.val)
                if node.left:   queue.append(node.left)
                if node.right:  queue.append(node.right)
            res.append(temp)
        return res
```

#### 32-III. 从上到下打印二叉树（中等）：按之字形逐层打印节点

请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

例如:
给定二叉树: `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回其层次遍历结果：

```
[
  [3],
  [20,9],
  [15,7]
]
```

**提示：**

1. `节点总数 <= 1000`

##### 解法一：层次遍历

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        res = []
        from collections import deque	# 使用双端队列
        queue = deque([root])
        while queue:
            layers = deque()
            for _ in range(len(queue)):
                node = queue.popleft()
                if len(res)%2==1:
                    layers.appendleft(node.val)		# 这里实现倒序
                else:
                    layers.append(node.val)
                if node.left:   queue.append(node.left)
                if node.right:  queue.append(node.right)
            res.append(list(layers))	# 注意这里需要将deque对象转换为list
        return res
```

#### 55-I. 二叉树的深度

输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

例如：

给定二叉树 `[3,9,20,null,null,15,7]`，

```
    3
   / \
  9  20
    /  \
   15   7
```

返回它的最大深度 3 。

**提示：**

1. `节点总数 <= 10000`

##### 解法一：层次遍历

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        depth = 0
        from collections import deque
        queue = deque([root])
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:   queue.append(node.left)
                if node.right:  queue.append(node.right)
            depth+=1
        return depth
```

##### 解法二：递归（后序遍历DFS）

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        l = self.maxDepth(root.left)
        r = self.maxDepth(root.right)
        return max(l,r)+1
```

#### 55-II. 平衡二叉树

输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

**示例 1:**

给定二叉树 `[3,9,20,null,null,15,7]`

```
    3
   / \
  9  20
    /  \
   15   7
```

返回 `true` 。

**示例 2:**

给定二叉树 `[1,2,2,3,3,null,null,4,4]`

```
       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
```

返回 `false` 。

**限制：**

- `1 <= 树的结点个数 <= 10000`

#### 57-I. 和为s的两个数字

输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

 **示例 1：**

```
输入：nums = [2,7,11,15], target = 9
输出：[2,7] 或者 [7,2]
```

**示例 2：**

```
输入：nums = [10,26,30,31,47,60], target = 40
输出：[10,30] 或者 [30,10]
```

**限制：**

- `1 <= nums.length <= 10^5`
- `1 <= nums[i] <= 10^6`

##### 解法一：对撞双指针

```python
# Python3
class Solution:
    def twoSum(self, nums, target):
        l, r, sum = 0, len(nums)-1, 0
        while l<r:
            sum = nums[l] + nums[r]
            if sum == target:
                return [nums[l], nums[r]]
            elif sum < target:
                l+=1
            else:
                r-=1
        return []
```

```c++
// C++
#include <iostream>
#include <vector>
using namespace std;
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int l=0, r=nums.size()-1, sum=0;
        while(l<r){
            sum = nums[l]+nums[r];
            if(sum==target)     
                return {nums[l], nums[r]};
            else if(sum<target) 
                l+=1;
            else   
                r-=1;   // 这里强行让C++代码风格和Python风格匹配，哈哈哈
        }
        return {};
    }
};
```



