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

#### 07. 重建二叉树（TBC）

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

##### 解法一：递归（TBC）

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

#### 11. 旋转数组的最小数字

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 `[3,4,5,1,2]` 为 `[1,2,3,4,5]` 的一个旋转，该数组的最小值为1。 

**示例 1：**

```
输入：[3,4,5,1,2]
输出：1
```

**示例 2：**

```
输入：[2,2,2,0,1]
输出：0
```

##### 解法一：二分查找

```python
# Python3
from typing import List
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        N = len(numbers)
        l, r = 0, N-1
        while l<r:
            mid = l+(r-l)//2
            if numbers[mid] < numbers[r]:
                r = mid
            elif numbers[mid] > numbers[r]:
                l = mid+1
            else:
                r-=1
        return numbers[l]
```

```c++
// C++
#include <iostream>
#include <vector>
using namespace std;
class Solution {
public:
    int minArray(vector<int>& numbers) {
        int N = numbers.size();
        int l=0, r=N-1;
        while(l<r){
            int mid = l + (r-l)/2;
            if(numbers[mid]<numbers[r])
                r = mid;
            else if(numbers[mid]>numbers[r])
                l = mid+1;
            else
                r-=1;
        }
        return numbers[l];
    }
};
```

#### 12. 矩阵中的路径（TBC）

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。

[["a","**b**","c","e"],
["s","**f**","**c**","s"],
["a","d","**e**","e"]]

但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。

 **示例 1：**

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```

**示例 2：**

```
输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false
```

 **提示：**

- `1 <= board.length <= 200`
- `1 <= board[i].length <= 200`

#### 13. 机器人的运动范围（TBC）

地上有一个m行n列的方格，从坐标 `[0,0]` 到坐标 `[m-1,n-1]` 。一个机器人从坐标 `[0, 0] `的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

 **示例 1：**

```
输入：m = 2, n = 3, k = 1
输出：3
```

**示例 2：**

```
输入：m = 3, n = 1, k = 0
输出：1
```

**提示：**

- `1 <= n,m <= 100`
- `0 <= k <= 20`

#### 14. I剪绳子

给你一根长度为 `n` 的绳子，请把绳子剪成整数长度的 `m` 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 `k[0],k[1]...k[m-1]` 。请问 `k[0]*k[1]*...*k[m-1]` 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

**示例 1：**

```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
```

**示例 2:**

```
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
```

**提示：**

- `2 <= n <= 58`

##### 解法一：动态规划

```python
# Python3
class Solution:
    def cuttingRope(self, n: int) -> int:

        if n==2:    return 1
        if n==3:    return 2
        dp = [1]*(n+1)
        dp[0] = 1
        dp[1] = 1
        dp[2] = 2
        dp[3] = 3       # 注意dp[3] != dp[1]*dp[2]，所以dp[3]必须指定初始值
        # 事实上，尽可能将绳子以长度3等分为多段时，乘积最大

        for i in range(4, n+1):
            for j in range(1, i//2+1):
                print(i, j, i-j)
                dp[i] = max(dp[j]*dp[i-j], dp[i])
        return dp[n]
```

```c++
//C++
#include<iostream>
#include<vector>
using namespace std;
class Solution{
public:
    int cuttingRope(int n){
        if(n==2)    return 1;
        if(n==3)    return 2;
        vector<int> dp(n+1, 1);
        dp[0] = 1;
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 3;  // 注意dp[3] != dp[1]*dp[2]，所以dp[3]必须指定初始值
        // 事实上，尽可能将绳子以长度3等分为多段时，乘积最大
        
        for(int i=4; i<n+1; i++){
            for(int j=1; j<i/2+1; j++){
                dp[i] = max(dp[j]*dp[i-j], dp[i]);
            }
        }
        return dp[n];
    }
};
```

#### 14. II剪绳子

给你一根长度为 `n` 的绳子，请把绳子剪成整数长度的 `m` 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 `k[0],k[1]...k[m - 1]` 。请问 `k[0]*k[1]*...*k[m - 1]` 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

 **示例 1：**

```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
```

**示例 2:**

```
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
```

 **提示：**

- `2 <= n <= 1000`

##### 解法二：贪心

```python
# Python3
class Solution:
    def cuttingRope(self, n: int) -> int:
        if n==2:    return 1
        if n==3:    return 2
        res = 1
        while(n>4):
            res*=3
            res%=1000000007
            n-=3
        return (res*n)%1000000007
```

```c++
// C++
#include<iostream>
using namespace std;
class Solution{
public:
    int cuttingRope(int n){
        if(n==2)    return 1;
        if(n==3)    return 2;
        long long res=1;
        while(n>4){
            res*=3;
            res%=1000000007;
            n-=3;
        }
        return (res*n)%1000000007;
    }
};
```

#### 15. 二进制中1的个数

请实现一个函数，输入一个整数（以二进制串形式），输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。

 **示例 1：**

```
输入：00000000000000000000000000001011
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
```

**示例 2：**

```
输入：00000000000000000000000010000000
输出：1
解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
```

**示例 3：**

```
输入：11111111111111111111111111111101
输出：31
解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。
```

 **提示：**

- 输入必须是长度为 `32` 的 **二进制串** 。

##### 解法一：逐位判断

```python
# Python
class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            res+= n&1
            n>>=1
        return res
```

```c++
// C++
#include<iostream>
using namespace std;
class Solution{
public:
    int hammingWeight(uint32_t n){
        int res = 0;
        while(n){
            res+= n&1;
            n>>=1;
        }
        return res;
    }
};
```

##### 解法二：巧用n&(n-1)

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            res+=1
            n&=(n-1)    # 次操作能消去n的二进制形式里最低位的那个1
        return res
```

```c++
class Solution{
public:
    int hammingWeight(uint32_t n){
        int res = 0;
        while(n){
            res+=1;
            n&=(n-1);
        }
        return res;
    }
};
```

#### 16. 数值的整数次方

实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数（即，$x^n$）。不得使用库函数，同时不需要考虑大数问题。

 **示例 1：**

```
输入：x = 2.00000, n = 10
输出：1024.00000
```

**示例 2：**

```
输入：x = 2.10000, n = 3
输出：9.26100
```

**示例 3：**

```
输入：x = 2.00000, n = -2
输出：0.25000
解释：2-2 = 1/22 = 1/4 = 0.25
```

 **提示：**

- `-100.0 < x < 100.0`
- `-231 <= n <= 231-1`
- `-104 <= xn <= 104`

##### 解法一：快速幂（循环解法）

```python
# Python3
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x==0:    return 0
        if n<0:
            x, n = 1/x, -n
        res = 1
        while n:
            if n&1:
                res*=x
            x*=x
            n>>=1
        return res
```

```c++
// C++
#include<iostream>
using namespace std;
class Solution{
public:
    double myPow(double x, int n){
        if(x==0)    return 0;
        double res = 1;
        long num = n;
        if(num<0){
            x = 1/x;
            num = -num;
        }
        while(num>0){   // C++里面，建议用num>0作为判断条件
            if(num&1) res*=x;
            x*=x;
            num>>=1;
        }
        return res;
    }
};
```

##### 解法二：快速幂（递归解法）

```python
# Python3
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x==0:    return 0
        if n<0:
            x, n = 1/x, -n
        if n==0:
            return 1    # 递归的停止条件
        if n&1:
            return self.myPow(x*x, n>>1) * x
        else:
            return self.myPow(x*x, n>>1)
```

```c++
// C++
#include<iostream>
using namespace std;
class Solution{
public:
    double myPow(double x, int n){
        if(x==0)    return 0;
        long num = n;
        if(num<0){
            x = 1/x;
            num = -num;
        }
        if(num==0)      // 递归的终止条件
            return 1;
        if(num&1==1)    
            return myPow(x*x, num>>1) * x;
        else
            return myPow(x*x, num>>1);
    }
};
```

#### 17. 打印从1到最大的n位数

输入数字 `n`，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

**示例 1:**

```
输入: n = 1
输出: [1,2,3,4,5,6,7,8,9]
```

 说明：

- 用返回一个整数列表来代替打印
- n 为正整数

##### 解法一：简单解法（不考虑大数问题）

```python
# Python3
class Solution:
    def printNumbers(self, n: int):
        return list(range(1, 10**n))
```

```c++
// C++
#include<iostream>
#include<vector>
#include<math.h>
using namespace std;
class Solution {
public:
    vector<int> printNumbers(int n) {
        vector<int> res;
        for(int i=1; i<pow(10, n); i++)
            res.push_back(i);
        return res;
    }
};
```

##### 解法二：考虑大数问题（TBC）

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

##### 解法一：单指针

```python
# Python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if head.val==val:   return head.next    # 如果头节点是要删除的节点
        root = head
        while head.next:
            if head.next.val==val:  # 找到待删除节点，将待删除节点的前驱节点的next指针指向待删除节点的next指针指向的节点（注意，当待删除节点是该链表最后一个节点的时候，该操作后待删除节点的前驱节点的next指针指向为None，则不能进行head = head.next操作，故需要break）
                head.next = head.next.next
                break     # 必须要有break，以应对删除的节点是最后一个节点的情况
            head = head.next
        return root     # 返回头结点指针
```

```c++
// C++
#include<iostream>
using namespace std;
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
// 单指针
class Solution {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        if(head->val == val)    return head->next;
        ListNode *root = head;
        while(head->next){
            if(head->next->val == val){
                head->next = head->next->next;
                break;
            }
            head = head->next;
        }
        return root;
    }
};
```

##### 解法二：双指针

```python
# Python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if head.val==val:   return head.next    # 如果头节点是要删除的节点
        root = head
        pre, cur = head, head.next  # 定义两个指针，一个指向前驱节点，一个指向当前节点
        while cur:
            if cur.val==val:    # 当找到要删除的节点的时候，将前驱节点的next指针指向当前节点的下个节点
                pre.next = cur.next
                break   # 可以不用这个break
            pre, cur = cur, cur.next
        return root     # 返回头结点指针
```

```c++
// C++
#include<iostream>
using namespace std;
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
class Solution {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        if(head->val == val)    return head->next;
        ListNode *root = head, *pre = head, *cur = head->next;
        while(cur){
            if(cur->val == val) 
                pre->next = cur->next;
            pre = cur;
            cur = cur->next;
        }
        return root;
    }
};
```

#### 19. 正则表达式匹配（TBC）

请实现一个函数用来匹配包含`'. '`和`'*'`的正则表达式。模式中的字符`'.'`表示任意一个字符，而`'*'`表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串`"aaa"`与模式`"a.a"`和`"ab*ac*a"`匹配，但与`"aa.a"`和`"ab*a"`均不匹配。

**示例 1:**

```
输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。
```

**示例 2:**

```
输入:
s = "aa"
p = "a*"
输出: true
解释: 因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
```

**示例 3:**

```
输入:
s = "ab"
p = ".*"
输出: true
解释: ".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
```

**示例 4:**

```
输入:
s = "aab"
p = "c*a*b"
输出: true
解释: 因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
```

**示例 5:**

```
输入:
s = "mississippi"
p = "mis*is*p*."
输出: false
```

- `s` 可能为空，且只包含从 `a-z` 的小写字母。
- `p` 可能为空，且只包含从 `a-z` 的小写字母以及字符 `.` 和 `*`，无连续的 `'*'`。

#### 20. 表示数值的字符串（TBC）

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"-1E-16"、"0123"都表示数值，但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是。

#### 21. 调整数组顺序使奇数位于偶数前面

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

 **示例：**

```
输入：nums = [1,2,3,4]
输出：[1,3,2,4] 
注：[3,1,2,4] 也是正确的答案之一。
```

 **提示：**

1. `0 <= nums.length <= 50000`
2. `1 <= nums[i] <= 10000`

##### 解法一：首尾双指针

```python
# Python3
class Solution:
    def exchange(self, nums):
        N = len(nums)
        if N<=1:    return nums # 空列表或者列表只有一个元素
        l, r = 0, N-1
        while l<r:
            while l<r:
                if nums[l]&1==0:    break    # 找到左边的偶数
                l+=1
            while l<r:
                if nums[r]&1==1:    break   # 找到右边的奇数
                r-=1
            nums[l], nums[r] = nums[r], nums[l]
        return nums
```

```C++
// C++
#include<iostream>
#include<vector>
using namespace std;
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        int N = nums.size();
        if(N<=1)    return nums;    // 空列表或者列表只有一个元素
        int l=0, r=N-1;
        while(l<r){
            while(l<r){
                if((nums[l]&1)==0) break; // 找到左边的偶数
                l+=1;
            }
            while(l<r){
                if((nums[r]&1)==1) break; // 找到右边的奇数
                r-=1;
            }
            swap(nums[r], nums[l]);
        }
        return nums;
    }
};
```

#### 22. 链表中倒数第k各节点（简单）

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。

**示例：**

```
给定一个链表: 1->2->3->4->5, 和 k = 2.

返回链表 4->5.
```

##### 解法一：快慢双指针（固定间隔）

```python
# Python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        left = right = head
        for _ in range(k):
            right = right.next
        while right:
            left, right = left.next, right.next
        return left
```

```c++
// C++
#include<iostream>
using namespace std;
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode *left=head, *right=head;
        for(int i=0; i<k; i++)
            right = right->next;
        while(right){
            left = left->next;
            right = right->next;
        }
        return left;
    }
};
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

##### 解法一：双指针（迭代）

```python
# Python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        pre, cur = None, head
        while cur:
            temp = cur.next         # 暂存后继节点，便于向右遍历
            cur.next = pre          # 修改当前节点的next指针指向，即将当前节点的next指针指向其前驱节点，完成反向操作
            pre, cur = cur, temp    # 向后遍历
        return pre
```

```c++
// C++
#include<iostream>
using namespace std;
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(head==nullptr || head->next==nullptr)
            return head;
        ListNode *pre=nullptr, *cur=head;
        while(cur){
            ListNode *temp = cur->next; // 暂存后继节点，便于向后遍历
            cur->next = pre;            // 修改当前节点的next指针指向，即将当前节点的next指针指向其前驱节点，完成反向操作
            pre = cur;                  // 向后遍历
            cur = temp;
        }
        return pre;
    }
};
```

##### 解法二：递归（TBC）

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
# Python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(0)
        head = dummy
        while l1 and l2:
            if l1.val < l2.val:
                head.next = l1
                l1 = l1.next
            else:
                head.next = l2
                l2 = l2.next
            head = head.next
        head.next = l1 if l1 else l2
        return dummy.next
```

```c++
// C++
#include<iostream>
using namespace std;
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode *dummy = new ListNode(0);
        ListNode *head = dummy;
        while(l1 && l2){
            if(l1->val < l2->val){
                head->next = l1;
                l1 = l1->next;
            }
            else{
                head->next = l2;
                l2 = l2->next;
            }
            head = head->next;
        }
        head->next = l1?l1:l2;
        return dummy->next;
    }
};
```

##### 解法二：递归

```python  
# Python3
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

```c++
// C++
#include<iostream>
using namespace std;
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1==nullptr)
            return l2;
        else if(l2==nullptr)
            return l1;
        else{
            if(l1->val < l2->val){
                l1->next = mergeTwoLists(l1->next, l2);
                return l1;
            }
            else{
                l2->next = mergeTwoLists(l1, l2->next);
                return l2;
            }
        }
    }
};
```

#### 26. 树的子结构（TBC）

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:

`   3  / \  4  5 / \ 1  2`
给定的树 B：

`  4  / 1`
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

**示例 1：**

```
输入：A = [1,2,3], B = [3,1]
输出：false
```

**示例 2：**

```
输入：A = [3,4,5,1,2], B = [4,1]
输出：true
```

**限制：**

```
0 <= 节点个数 <= 10000
```

#### 27. 二叉树的镜像（TBC）

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

例如输入：

 	4  

​    /  \ 

   2   7 

  / \  / \

1  3 6  9
镜像输出：

```
    4 
  /  \ 
 7   2 
/ \  / \
9  6 3  1
```

 **示例 1：**

```
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```

 **限制：**

```
0 <= 节点个数 <= 1000
```

##### 解法一：辅助栈（TBC-C++）

```python
class Solution:
    def mirrorTree(self, root):
        """
        辅助栈或者辅助队列均可，本质为遍历这棵树所有的节点，然后交换pop出的节点的左右节点
        """
        if not root:
            return root
        stack = []
        stack.append(root)
        while stack:
            node = stack.pop(0)
            node.left, node.right = node.right, node.left
            if node.left:       stack.append(node.left)
            if node.right:      stack.append(node.right)
        return root
```



#### 28. 对称的二叉树（TBC）

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

​	1  

​    / \ 

   2  2 

  / \ / \

3  4 4  3
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

```
  1  
 / \ 
 2  2 
  \  \ 
   3  3
```

 **示例 1：**

```
输入：root = [1,2,2,3,4,4,3]
输出：true
```

**示例 2：**

```
输入：root = [1,2,2,null,3,null,3]
输出：false
```

 **限制：**

```
0 <= 节点个数 <= 1000
```

#### 29. 顺时针打印矩阵

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

**示例 1：**

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

**示例 2：**

```
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
```

 **限制：**

- `0 <= matrix.length <= 100`
- `0 <= matrix[i].length <= 100`

##### 解法一：按层模拟

```python
# Python3
class Solution:
    def spiralOrder(self, matrix):
        res = []
        rows, cols = (len(matrix) if matrix else 0), (len(matrix[0]) if matrix else 0)
        if rows==0 or cols==0:
            return res
        left, right, top, bottom = 0, cols-1, 0, rows-1
        while left<=right and top<=bottom:
            for col in range(left, right+1):
                res.append(matrix[top][col])
            for row in range(top+1, bottom+1):
                res.append(matrix[row][right])

            if left<right and top<bottom:
                for col in range(right-1, left, -1):
                    res.append(matrix[bottom][col])
                for row in range(bottom, top, -1):
                    res.append(matrix[row][left])
            left, right, top, bottom = left+1, right-1, top+1, bottom-1
        return res
```

```c++
// C++
#include<iostream>
#include<vector>
using namespace std;
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int rows=matrix.empty()?0:matrix.size(), cols=matrix.empty()?0:matrix[0].size();
        vector<int> res;
        if(rows==0 || cols==0)  
            return res;
        int left=0, right=cols-1, top=0, bottom=rows-1;
        while((left<=right) && (top<=bottom)){
            for(int col=left; col<right+1; col++)
                res.push_back(matrix[top][col]);
            for(int row=top+1; row<bottom+1; row++)
                res.push_back(matrix[row][right]);
            if((left<right) && (top<bottom)){
                for(int col=right-1; col>left; col--)
                    res.push_back(matrix[bottom][col]);
                for(int row=bottom; row>top; row--)
                    res.push_back(matrix[row][left]);
            }
            left++; right--; top++; bottom--;
        }
        return res;
    }
};
```

#### 30. 包含min函数的栈（TBC）

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

**示例:**

```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
```

**提示：**

1. 各函数的调用总次数不超过 20000 次

#### 31. 栈的压入、弹出序列（TBC）

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

**示例 1：**

```
输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
```

**示例 2：**

```
输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。
```

 **提示：**

1. `0 <= pushed.length == popped.length <= 1000`
2. `0 <= pushed[i], popped[i] < 1000`
3. `pushed` 是 `popped` 的排列。

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
# Python3
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from typing import List
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        valArray = []
        if not root:
            return valArray

        from collections import deque
        queue = deque([root])
        while queue:
            for _ in range(len(queue)):
                curNode = queue.popleft()
                valArray.append(curNode.val)

                if curNode.left:    queue.append(curNode.left)
                if curNode.right:   queue.append(curNode.right)

        return valArray
```

```c++
// C++
#include<iostream>
#include<vector>
#include<deque>
using namespace std;
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution {
public:
    vector<int> levelOrder(TreeNode* root) {
        vector<int> valArray;
        if(root==nullptr)   return valArray;
        deque<TreeNode*> queue={root};
        while(!queue.empty()){
            int length = queue.size();
            for(int i=0; i<length; i++){
                TreeNode *curNode = queue.front();
                queue.pop_front();
                valArray.push_back(curNode->val);

                if(curNode->left)   queue.push_back(curNode->left);
                if(curNode->right)  queue.push_back(curNode->right);
            }
        }
        return valArray;
    }
};
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
# Python3
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from typing import List
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        valArray = []
        if not root:
            return valArray

        from collections import deque
        queue = deque()
        queue.append(root)
        while(queue):
            length = len(queue)
            valLayer = []
            for _ in range(length):
                curNode = queue.popleft()
                valLayer.append(curNode.val)

                if curNode.left:    queue.append(curNode.left)
                if curNode.right:   queue.append(curNode.right)
            valArray.append(valLayer)

        return valArray
```

```c++
// C++
#include<iostream>
#include<vector>
#include<deque>
using namespace std;
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> valArray;
        if(root==nullptr)   return valArray;
        deque<TreeNode*> queue={root};
        while(!queue.empty()){
            int length = queue.size();
            vector<int> valLayer;
            for(int i=0; i<length; i++){
                TreeNode *curNode = queue.front();
                queue.pop_front();
                valLayer.push_back(curNode->val);

                if(curNode->left)   queue.push_back(curNode->left);
                if(curNode->right)  queue.push_back(curNode->right);
            }
            valArray.push_back(valLayer);
        }
        return valArray;
    }
};
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
# Python3
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from typing import List
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        valArray = []
        if not root:
            return valArray

        from collections import deque
        queue = deque()
        queue.append(root)
        layerIdx = -1
        while queue:
            length = len(queue)
            valLayer = []
            layerIdx+=1
            for _ in range(length):
                curNode = queue.popleft()
                valLayer.append(curNode.val)

                if curNode.left:    queue.append(curNode.left)
                if curNode.right:   queue.append(curNode.right)

            if layerIdx&1==1:
                valLayer.reverse()
            valArray.append(valLayer)

        return valArray
```

```c++
// C++
#include<iostream>
#include<vector>
#include<deque>
#include<algorithm>
using namespace std;
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> valArray;
        if(root==nullptr)   return valArray;
        
        deque<TreeNode*> queue;
        queue.push_back(root);
        int layerIdx = -1;
        while(!queue.empty()){
            int length = queue.size();
            vector<int> valLayer;
            layerIdx+=1;
            for(int i=0; i<length; i++){
                TreeNode *curNode = queue.front();
                queue.pop_front();
                valLayer.push_back(curNode->val);

                if(curNode->left)   queue.push_back(curNode->left);
                if(curNode->right)  queue.push_back(curNode->right);
            }
            if(layerIdx&1==1)
                reverse(valLayer.begin(), valLayer.end());
            valArray.push_back(valLayer);
        }
        
        return valArray;
    }
};
```

#### 33. 二叉搜索树的后序遍历序列

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 `true`，否则返回 `false`。假设输入的数组的任意两个数字都互不相同。

 参考以下这颗二叉搜索树：

```
     5
    / \
   2   6
  / \
 1   3
```

**示例 1：**

```
输入: [1,6,3,2,5]
输出: false
```

**示例 2：**

```
输入: [1,3,2,6,5]
输出: true
```

 **提示：**

1. `数组长度 <= 1000`

##### 解法一：递归分治

```python
# Python3
class Solution:
    def verifyPostorder(self, postorder):
        def recur(nums):
            if len(nums)<=1:    return True

            l, r = 0, len(nums)-1
            while nums[l]<nums[r]:     l+=1
            leftTree = nums[0:l]
            m = l
            while nums[l]>nums[r]:     l+=1
            rightTree= nums[m:l]
            return l==r and recur(leftTree) and recur(rightTree)
        return recur(postorder)
```

```c++
// C++
#include<iostream> 
#include<vector>
using namespace std;
class Solution {
public:
    bool recur(vector<int>& nums){
        if(nums.size()<=1)  return true;
        int l=0, r=nums.size()-1;
        vector<int> leftTree;
        while(nums[l]<nums[r]){
            leftTree.push_back(nums[l]);
            l+=1;
        }
        vector<int> rightTree;
        while(nums[l]>nums[r]){
            rightTree.push_back(nums[l]);
            l+=1;
        }
        return (l==r) && recur(leftTree) && recur(rightTree);
    }
    bool verifyPostorder(vector<int>& postorder) {
        return recur(postorder);
    }
};
```

##### 解法二：辅助单调栈（TBC）

#### 35. 复杂链表的复制（TBC）

请实现 `copyRandomList` 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 `next` 指针指向下一个节点，还有一个 `random` 指针指向链表中的任意节点或者 `null`。

 **示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e1.png)

```
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
```

**示例 2：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e2.png)

```
输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]
```

**示例 3：**

**![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e3.png)**

```
输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]
```

**示例 4：**

```
输入：head = []
输出：[]
解释：给定的链表为空（空指针），因此返回 null。
```

 **提示：**

- `-10000 <= Node.val <= 10000`
- `Node.random` 为空（null）或指向链表中的节点。
- 节点数目不超过 1000 。

#### 36. 二叉搜索树与双向链表（TBC）

#### 37. 序列化二叉树（TBC）

请实现两个函数，分别用来序列化和反序列化二叉树。

**示例:** 

```
你可以将以下二叉树：

    1
   / \
  2   3
     / \
    4   5

序列化为 "[1,2,3,null,null,4,5]"
```

#### 38. 字符串的排列（TBC）

输入一个字符串，打印出该字符串中字符的所有排列。

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

**示例:**

```
输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]
```

**限制：**

```
1 <= s 的长度 <= 8
```

#### 39. 数组中出现次数超过一半的数字

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

**示例 1:**

```
输入: [1, 2, 3, 2, 2, 2, 5, 4, 2]
输出: 2
```

**示例 2：**

```
输入：[2,2,1,1,1,2,2]
输出：2
```

**限制：**

```
1 <= 数组长度 <= 50000
```

**进阶：**

- 尝试设计时间复杂度为 O(n)、空间复杂度为 O(1) 的算法解决此问题。

##### 解法一：哈希表

```python
# Python3
class Solution:
    def majorityElement(self, nums):
        from collections import defaultdict
        dic = defaultdict(int)
        N = len(nums)
        for num in nums:
            dic[num]+=1
            if dic[num]>N//2:
                return num
        return nums[0]
```

```c++
// C++
#include<iostream>
#include<vector>
#include<unordered_map>
using namespace std;
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        unordered_map<int, int> dic;
        int N = nums.size();
        for(auto num:nums){
            dic[num]+=1;
            if(dic[num]>N/2)
                return num;
        }
        return nums[0];
    }
};
```

##### 解法二：摩尔投票

```python
# Python3
class Solution:
    def majorityElement(self, nums):
        vote = 0
        mode = None
        for num in nums:
            if vote==0:
                mode = num

            if mode==num:
                vote+=1
            else:
                vote-=1
        return mode
```

```c++
// C++
#include<iostream>
#include<vector>
#include<unordered_map>
using namespace std;
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int vote=0, mode;
        for(auto num:nums){
            if(vote==0)
                mode = num;
            
            if(mode==num)
                vote+=1;
            else
                vote-=1;
        }
        return mode;
    }
};
```

#### 40. 最小的k个数

输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

 **示例 1：**

```
输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
```

**示例 2：**

```
输入：arr = [0,1,2,1], k = 1
输出：[0]
```

 **限制：**

- `0 <= k <= arr.length <= 10000`
- `0 <= arr[i] <= 10000`

##### 解法一：排序

```python
# Python3
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        return sorted(arr)[:k]
```

```c++
// C++
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        sort(arr.begin(), arr.end());
        vector<int> res(k, 0);
        for(int i=0; i<k; i++){
            res[i] = arr[i];
        }
        return res;
    }
};
```

##### 解法二：堆

```python
# Python3
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        res = []
        if k==0:    return res

        import heapq
        newArr = [-x for x in arr]
        que = newArr[:k]
        heapq.heapify(que)
        for i in range(k, len(newArr)):
            if newArr[i]>que[0]:
                heapq.heappop(que)
                heapq.heappush(que, newArr[i])
        res = [-x for x in que]
        return res
```

```c++
// C++
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int> res(k, 0);
        if(k==0)    return res;
        priority_queue<int> que;
        for(int i=0; i<k; i++)
            que.push(arr[i]);
        for(int i=k; i<arr.size(); i++){
            if(arr[i]<que.top()){
                que.pop();
                que.push(arr[i]);
            }
        }
        for(int i=0; i<k; i++){
            res[i] = que.top();
            que.pop();
        }
        return res;
    }
};
```

##### 解法三：快排思想（TBC）

#### 41. 数据流中的中位数（TBC）

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

例如，

[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：

- void addNum(int num) - 从数据流中添加一个整数到数据结构中。
- double findMedian() - 返回目前所有元素的中位数。

**示例 1：**

```
输入：
["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]
[[],[1],[2],[],[3],[]]
输出：[null,null,null,1.50000,null,2.00000]
```

**示例 2：**

```
输入：
["MedianFinder","addNum","findMedian","addNum","findMedian"]
[[],[2],[],[3],[]]
输出：[null,null,2.00000,null,2.50000]
```

 **限制：**

- 最多会对 `addNum、findMedian` 进行 `50000` 次调用。

#### 42. 连续子数组的最大和

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

 **示例1:**

```
输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

**提示：**

- `1 <= arr.length <= 10^5`
- `-100 <= arr[i] <= 100`

##### 解法一：动态规划

```python
# Python3
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        N = len(nums)
        dp = [0] * N  # 定义dp数组
        dp[0] = nums[0]  # 初始值
        max_sum = dp[0]  # 记录遍历到的具有最大和的子数组之和
        for i in range(1, N):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
            max_sum = max(max_sum, dp[i])
        return max_sum
```

```c++
// C++
#include<iostream>
#include<vector>
using namespace std;
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int N=nums.size();
        vector<int> dp(N, 0);
        dp[0] = nums[0];
        int max_sum = dp[0];
        for(int i=1; i<N; i++){
            dp[i] = max(dp[i-1]+nums[i], nums[i]);
            max_sum = max(max_sum, dp[i]);
        }
        return max_sum;
    }
};
```

#### 43. 1～n 整数中 1 出现的次数（TBC）

输入一个整数 `n` ，求1～n这n个整数的十进制表示中1出现的次数。

例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

 **示例 1：**

```
输入：n = 12
输出：5
```

**示例 2：**

```
输入：n = 13
输出：6
```

 **限制：**

- `1 <= n < 2^31`

#### 44. 数字序列中某一位的数字（TBC）

数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。

请写一个函数，求任意第n位对应的数字。

 **示例 1：**

```
输入：n = 3
输出：3
```

**示例 2：**

```
输入：n = 11
输出：0
```

 **限制：**

- `0 <= n < 2^31`

#### 45. 把数组排成最小的数

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

**示例 1:**

```
输入: [10,2]
输出: "102"
```

**示例 2:**

```
输入: [3,30,34,5,9]
输出: "3033459"
```

 **提示:**

- `0 < nums.length <= 100`

**说明:**

- 输出结果可能非常大，所以你需要返回一个字符串而不是整数
- 拼接起来的数字可能会有前导 0，最后结果不需要去掉前导 0

#### 46. 把数字翻译成字符串

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

**示例 1:**

```
输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
```

**提示：**

- `0 <= num < 231`

#### 47. 礼物的最大价值

在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

**示例 1:**

```
输入: 
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
```

提示：

- `0 < grid.length <= 200`
- `0 < grid[0].length <= 200`

```python
# Python3
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        M, N = len(grid), len(grid[0])
        dp = [[0] * N for _ in range(M)]
        dp[0][0] = grid[0][0]
        for x in range(1, M):
            dp[x][0] = grid[x][0] + dp[x - 1][0]
        for y in range(1, N):
            dp[0][y] = grid[0][y] + dp[0][y - 1]

        for x in range(1, M):
            for y in range(1, N):
                dp[x][y] = grid[x][y] + max(dp[x - 1][y], dp[x][y - 1])
        return dp[-1][-1]
```

```c++
// C++
#include<iostream>
#include<vector>
using namespace std;
class Solution {
public:
    int maxValue(vector<vector<int>>& grid) {
        if(grid.empty() || grid[0].empty())    return 0;
        int M=grid.size(), N=grid[0].size();
        vector<vector<int>> dp(M, vector<int>(N, 0));
        dp[0][0] = grid[0][0];
        for(int x=1; x<M; x++)  dp[x][0] = grid[x][0]+dp[x-1][0];
        for(int y=1; y<N; y++)  dp[0][y] = grid[0][y]+dp[0][y-1];
        for(int x=1; x<M; x++){
            for(int y=1; y<N; y++){
                dp[x][y] = grid[x][y]+max(dp[x-1][y], dp[x][y-1]);
            }
        }
        return dp.back().back();
    }
};
```

#### 48. 最长不含重复字符的子字符串（TBC）

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

**示例 1:**

```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**示例 2:**

```
输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

**示例 3:**

```
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

提示：

- `s.length <= 40000`

#### 49. 丑数（TBC）

我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

**示例:**

```
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
```

**说明:** 

1. `1` 是丑数。
2. `n` **不超过**1690。

#### 50. 第一个只出现一次的字符（TBC）

在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

**示例:**

```
s = "abaccdeff"
返回 "b"

s = "" 
返回 " "
```

 **限制：**

```
0 <= s 的长度 <= 50000
```

#### 51. 数组中的逆序对（TBC）

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

 **示例 1:**

```
输入: [7,5,6,4]
输出: 5
```

 **限制：**

```
0 <= 数组长度 <= 50000
```

#### 52. 两个链表的第一个公共节点（TBC）

输入两个链表，找出它们的第一个公共节点。

如下面的两个链表**：**

[<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png" alt="img" style="zoom:50%;" />](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)

在节点 c1 开始相交。

 **示例 1：**

[<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_1.png" alt="img" style="zoom: 50%;" />](https://assets.leetcode.com/uploads/2018/12/13/160_example_1.png)

```
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
```

 **示例 2：**

[<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_2.png" alt="img" style="zoom: 50%;" />](https://assets.leetcode.com/uploads/2018/12/13/160_example_2.png)

```
输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Reference of the node with value = 2
输入解释：相交节点的值为 2 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。
```

 **示例 3：**

[<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_3.png" alt="img" style="zoom: 50%;" />](https://assets.leetcode.com/uploads/2018/12/13/160_example_3.png)

```
输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：null
输入解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
解释：这两个链表不相交，因此返回 null。
```

 **注意：**

- 如果两个链表没有交点，返回 `null`.
- 在返回结果后，两个链表仍须保持原有的结构。
- 可假定整个链表结构中没有循环。
- 程序尽量满足 O(*n*) 时间复杂度，且仅用 O(*1*) 内存。

#### 53-I. 在排序数组中查找数字数组中出现的次数（TBC）

 **示例 1:**

```
输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
```

**示例 2:**

```
输入: nums = [5,7,7,8,8,10], target = 6
输出: 0
```

 **限制：**

```
0 <= 数组长度 <= 50000
```

#### 53-II. 0～n-1中缺失的数字（TBC）

一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

 **示例 1:**

```
输入: [0,1,3]
输出: 2
```

**示例 2:**

```
输入: [0,1,2,3,4,5,6,7,9]
输出: 8
```

 **限制：**

```
1 <= 数组长度 <= 10000
```

#### 54. 二叉搜索树的第k大节点（TBC）

给定一棵二叉搜索树，请找出其中第k大的节点。

 **示例 1:**

```
输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 4
```

**示例 2:**

```
输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 4
```

 **限制：**

1 ≤ k ≤ 二叉搜索树元素个数

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

```c++
// C++
#include<iostream>
#include<deque>
using namespace std;
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(root==nullptr)   return 0;
        int depth=0;
        deque<TreeNode*> que = {root};
        while(!que.empty()){
            int length = que.size();
            for(int i=0; i<length; i++){
                TreeNode *curNode = que.front();
                que.pop_front();
                if(curNode->left)   que.push_back(curNode->left);
                if(curNode->right)  que.push_back(curNode->right);
            }
            depth+=1;
        }
        return depth;
    }
};
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

```c++
// C++
#include<iostream>
#include<deque>
using namespace std;
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(root==nullptr)   return 0;
        int l = maxDepth(root->left);
        int r = maxDepth(root->right);
        return max(l, r)+1;
    }
};
```



#### 55-II. 平衡二叉树（TBC）

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

#### 56-I. 数组中数字出现的次数（TBC）

一个整型数组 `nums` 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

**示例 1：**

```
输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]
```

**示例 2：**

```
输入：nums = [1,2,10,4,1,4,3,3]
输出：[2,10] 或 [10,2]
```

**限制：**

- `2 <= nums.length <= 10000`

#### 56-II. 数组中数字出现的次数（TBC）

在一个数组 `nums` 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

**示例 1：**

```
输入：nums = [3,4,3,3]
输出：4
```

**示例 2：**

```
输入：nums = [9,1,7,9,7,9,7]
输出：1
```

**限制：**

- `1 <= nums.length <= 10000`
- `1 <= nums[i] < 2^31`

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

#### 57-II. 和为s的连续正数序列（TBC）

输入一个正整数 `target` ，输出所有和为 `target` 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

**示例 1：**

```
输入：target = 9
输出：[[2,3,4],[4,5]]
```

**示例 2：**

```
输入：target = 15
输出：[[1,2,3,4,5],[4,5,6],[7,8]]
```

**限制：**

- `1 <= target <= 10^5`

#### 58-I. 翻转单词顺序（TBC）

输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

 **示例 1：**

```
输入: "the sky is blue"
输出: "blue is sky the"
```

**示例 2：**

```
输入: "  hello world!  "
输出: "world! hello"
解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
```

**示例 3：**

```
输入: "a good   example"
输出: "example good a"
解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
```

 **说明：**

- 无空格字符构成一个单词。
- 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
- 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。

#### 58-II. 左旋转字符串（TBC）

字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

**示例 1：**

```
输入: s = "abcdefg", k = 2
输出: "cdefgab"
```

**示例 2：**

```
输入: s = "lrloseumgh", k = 6
输出: "umghlrlose"
```

**限制：**

- `1 <= k < s.length <= 10000`

#### 59-I. 滑动窗口的最大值（TBC）

给定一个数组 `nums` 和滑动窗口的大小 `k`，请找出所有滑动窗口里的最大值。

**示例:**

```
输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7] 
解释: 

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

**提示：**

你可以假设 *k* 总是有效的，在输入数组不为空的情况下，1 ≤ k ≤ 输入数组的大小。

#### 59-II. 队列的最大值（TBC）

请定义一个队列并实现函数 `max_value` 得到队列里的最大值，要求函数`max_value`、`push_back` 和 `pop_front` 的**均摊**时间复杂度都是O(1)。

若队列为空，`pop_front` 和 `max_value` 需要返回 -1

**示例 1：**

```
输入: 
["MaxQueue","push_back","push_back","max_value","pop_front","max_value"]
[[],[1],[2],[],[],[]]
输出: [null,null,null,2,1,2]
```

**示例 2：**

```
输入: 
["MaxQueue","pop_front","max_value"]
[[],[],[]]
输出: [null,-1,-1]
```

**限制：**

- `1 <= push_back,pop_front,max_value的总操作数 <= 10000`
- `1 <= value <= 10^5`

#### 60. n个骰子的点数（TBC）

把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。

**示例 1:**

```
输入: 1
输出: [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]
```

**示例 2:**

```
输入: 2
输出: [0.02778,0.05556,0.08333,0.11111,0.13889,0.16667,0.13889,0.11111,0.08333,0.05556,0.02778]
```

**限制：**

```
1 <= n <= 11
```

#### 61. 扑克牌中的顺子（TBC）

从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

**示例 1:**

```
输入: [1,2,3,4,5]
输出: True
```

**示例 2:**

```
输入: [0,0,1,2,5]
输出: True
```

**限制：**

数组长度为 5 

数组的数取值为 [0, 13] .

#### 62. 圆圈中最后剩下的数字（TBC）

0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

**示例 1：**

```
输入: n = 5, m = 3
输出: 3
```

**示例 2：**

```
输入: n = 10, m = 17
输出: 2
```

**限制：**

- `1 <= n <= 10^5`
- `1 <= m <= 10^6`

#### 63. 股票的最大利润（TBC）

假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

**示例 1:**

```
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
```

**示例 2:**

```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```

**限制：**

```
0 <= 数组长度 <= 10^5求1+2+…+n
```

#### 64. 求1+2+…+n

求 `1+2+...+n` ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

**示例 1：**

```
输入: n = 3
输出: 6
```

**示例 2：**

```
输入: n = 9
输出: 45
```

**限制：**

- `1 <= n <= 10000`

##### 解法一：递归（使用逻辑运算符）

```python
class Solution:
    def sumNums(self, n: int) -> int:
        return n and n+self.sumNums(n-1)	# 使用 and
    	# return n==1 or n+self.sumNums(n-1)	# 使用 or
```

```c++
#include<iostream>
using namespace std;
class Solution {
public:
    int sumNums(int n) {
        n && (n+=sumNums(n-1));		// 使用 &&
        // n==1 || (n+=sumNums(n-1));	//使用 ||
        return n;
    }
};
```

##### 解法二：快速乘（TBC）

#### 65. 不用加减乘除做加法（TBC）

写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

**示例:**

```
输入: a = 1, b = 1
输出: 2
```

**提示：**

- `a`, `b` 均可能是负数或 0
- 结果不会溢出 32 位整数

#### 66. 构建乘积数组（TBC）

给定一个数组 `A[0,1,…,n-1]`，请构建一个数组 `B[0,1,…,n-1]`，其中 `B[i]` 的值是数组 `A` 中除了下标 `i` 以外的元素的积, 即 `B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]`。不能使用除法。

**示例:**

```
输入: [1,2,3,4,5]
输出: [120,60,40,30,24]
```

**提示：**

- 所有元素乘积之和不会溢出 32 位整数
- `a.length <= 100000`

#### 67. 把字符串转换成整数（TBC）

写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似的库函数。

首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。

当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。

该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。

注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。

在任何情况下，若函数不能进行有效的转换时，请返回 0。

**说明：**

假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231, 231 − 1]。如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。

**示例 1:**

```
输入: "42"
输出: 42
```

**示例 2:**

```
输入: "   -42"
输出: -42
解释: 第一个非空白字符为 '-', 它是一个负号。
     我们尽可能将负号与后面所有连续出现的数字组合起来，最后得到 -42 。
```

**示例 3:**

```
输入: "4193 with words"
输出: 4193
解释: 转换截止于数字 '3' ，因为它的下一个字符不为数字。
```

**示例 4:**

```
输入: "words and 987"
输出: 0
解释: 第一个非空字符是 'w', 但它不是数字或正、负号。
     因此无法执行有效的转换。
```

**示例 5:**

```
输入: "-91283472332"
输出: -2147483648
解释: 数字 "-91283472332" 超过 32 位有符号整数范围。 
     因此返回 INT_MIN (−231) 。
```

#### 68-I. 二叉搜索树的最近公共祖先（TBC）

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

[百度百科](https://baike.baidu.com/item/最近公共祖先/8918834?fr=aladdin)中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

例如，给定如下二叉搜索树: root = [6,2,8,0,4,7,9,null,null,3,5]

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/binarysearchtree_improved.png)

 

**示例 1:**

```
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。
```

**示例 2:**

```
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
输出: 2
解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
```

**说明:**

- 所有节点的值都是唯一的。
- p、q 为不同节点且均存在于给定的二叉搜索树中。

#### 68-II. 二叉树的最近公共祖先（TBC）

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

[百度百科](https://baike.baidu.com/item/最近公共祖先/8918834?fr=aladdin)中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

例如，给定如下二叉树: root = [3,5,1,6,2,0,8,null,null,7,4]

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/15/binarytree.png)

**示例 1:**

```
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
```

**示例 2:**

```
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出: 5
解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。
```

**说明:**

- 所有节点的值都是唯一的。
- p、q 为不同节点且均存在于给定的二叉树中。

#### 面试题34. 二叉树中和为某一值的路径（TBC）

输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。

**示例:**
给定如下二叉树，以及目标和 `target = 22`，

```
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1
```

返回:

```
[
   [5,4,11,2],
   [5,8,4,5]
]
```

**提示：**

1. `节点总数 <= 10000`